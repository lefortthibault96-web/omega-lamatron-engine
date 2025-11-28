#!/usr/bin/env python3
import re
from pathlib import Path
from rich.console import Console
from config import (
    AUTO_SUMMARIZE,
    CONTEXT_THRESHOLD,
    FULL_TURNS_TO_KEEP,
    SCENE_CONTEXT_THRESHOLD,
    SUMMARY_TEMPERATURE,
    MIN_SUMMARY_TURNS_TO_KEEP,
    DYNAMIC_SUMMARY_ALLOW_MIDTURN,
    prompts_dir,
    DEFAULT_MODEL,
)
from utils import (
    DEFAULT_MODEL_TOKEN_LIMIT,
    TURN_HEADER_RE,
    SceneDocument,
    clean_scene_turns,
    parse_collapsed_turns,
    has_turn_in_progress,
    extract_description,
    count_tokens_for_text_blocks,
    join_blocks,
)
from ollama import generate
from Prompt_Manager2000 import PromptManager


console = Console()
pm = PromptManager(prompts_dir)


# ---------------------------------------------------------
# LLM WRAPPER (unchanged)
# ---------------------------------------------------------
def llm_generate_for_summary(messages: list[dict]) -> str:
    """
    Calls the LLM for summarization using a fixed model and temperature.
    """
    model = DEFAULT_MODEL
    temperature = SUMMARY_TEMPERATURE

    system_prompt = "\n".join(
        msg["content"] for msg in messages if msg["role"] == "system"
    ).strip()
    user_prompt = "\n".join(
        msg["content"] for msg in messages if msg["role"] == "user"
    ).strip()

    resp = generate(
        model=model,
        prompt=user_prompt,
        system=system_prompt,
        options={"temperature": temperature},
    )

    return resp["response"].strip()


# ---------------------------------------------------------
# CORE TURN OPERATIONS
# ---------------------------------------------------------
def ensure_current_turn(scene_path: Path):
    """
    Ensures scene contains:
      - # Description
      - at least one # Turn N
    Does NOT create a new turn header beyond Turn 1.
    """
    if not scene_path or not scene_path.exists():
        console.print("[red]No active scene file found.[/red]")
        return 0

    text = scene_path.read_text(encoding="utf-8")

    # Wrap with SceneDocument
    doc = SceneDocument(text, lambda x: 0)

    modified = False
    lines = text.splitlines()

    # Ensure # Description exists
    if not doc.description():
        console.print("[cyan]No # Description found — creating one…[/cyan]")
        lines = ["# Description", "", ""] + lines
        modified = True

    # Ensure first Turn exists
    if not doc.turns:
        console.print("[cyan]No turns found — creating # Turn 1…[/cyan]")
        lines.append("")
        lines.append("# Turn 1")
        lines.append("")
        modified = True

    if modified:
        new_text = "\n".join(lines)
        scene_path.write_text(new_text, encoding="utf-8")
        doc = SceneDocument(new_text, lambda x: 0)

    # Determine last existing turn #
    if doc.turns:
        last_turn = max(t["index"] for t in doc.turns)
    else:
        last_turn = 1

    return last_turn


def advance_turn(scene_path, agent, current_turn=None):
    """
    Create a new Turn N block.
    """
    if not scene_path or not scene_path.exists():
        console.print("[red]No active scene file found.[/red]")
        return

    text = scene_path.read_text(encoding="utf-8")
    doc = SceneDocument(text, lambda x: 0)

    last_turn = max((t["index"] for t in doc.turns), default=0)
    if current_turn is None:
        current_turn = last_turn

    new_turn = last_turn + 1
    console.print(f"[cyan]Advancing to Turn {new_turn}…[/cyan]")

    updated = text.rstrip() + f"\n\n# Turn {new_turn} (In Progress)\n"
    scene_path.write_text(updated, encoding="utf-8")

    agent.current_turn = new_turn


# ---------------------------------------------------------
# DYNAMIC SUMMARIZATION (refactored)
# ---------------------------------------------------------
def dynamic_summarize_scene(pm, batcher, agent, system_prompts=None,
                            allow_fulltext_collapse=None):
    """
    Build a dynamically-collapsed scene context for the LLM.

    Behavior:
    - Keeps at least FULL_TURNS_TO_KEEP full turns and MIN_SUMMARY_TURNS_TO_KEEP summarized.
    - Uses BatchManager.build_dynamic_summarization_batch() to decide which turns to include.
    - If the batcher reports dropped turns, it triggers full-text summarization
      (unless a Turn is (In Progress) and DYNAMIC_SUMMARY_ALLOW_MIDTURN is False).
    - Respects DYNAMIC_SUMMARY_ALLOW_MIDTURN: when False and a turn is in progress,
      we do NOT trigger summarize_turn_fulltext_batches and instead stop collapsing.
    """

    # number of turns we must always preserve:
    min_preserved = FULL_TURNS_TO_KEEP + MIN_SUMMARY_TURNS_TO_KEEP

    # mid-turn config
    if allow_fulltext_collapse is None:
        allow_fulltext_collapse = DYNAMIC_SUMMARY_ALLOW_MIDTURN

    # --- initial state ---
    scene_working = agent.read_active_scene()
    collapsed_turns = parse_collapsed_turns(scene_working)
    in_progress = has_turn_in_progress(scene_working)

    threshold_tokens = int(DEFAULT_MODEL_TOKEN_LIMIT * CONTEXT_THRESHOLD)

    # Build default system prompts if not provided
    if system_prompts is None:
        system_prompts = pm.build_single_character_messages(
            system_prompt=pm.system_prompt(),
            character_instructions=pm.character_instructions(),
            submode_instructions=pm.submode("roleplay") or "",
            character_sheet="",
            speaker_name="unused",
            scene_text="",
            user_input=""
        )

    # Count tokens used by system content
    system_tokens = sum(agent.count_tokens_string(m["content"]) for m in system_prompts)
    available_tokens = threshold_tokens - system_tokens
    if available_tokens <= 0:
        raise ValueError("System content too large to proceed.")

    collapse_passes = 0
    MAX_COLLAPSE_PASSES = 5

    candidate_text = ""
    included_summaries = []
    included_fulls = []

    # =====================================================
    # MAIN LOOP
    # =====================================================
    while True:
        candidate_text, included_summaries, included_fulls = batcher.build_dynamic_summarization_batch(
            scene_text=scene_working,
            threshold_tokens=available_tokens,
            ignored_turns=collapsed_turns,
            system_prompts=system_prompts,
        )

        # -------------------------------------------------
        # CASE A — DROPPED TURNS (batcher requests collapse)
        # -------------------------------------------------
        if candidate_text is None:
            # The batcher returns dropped turns as the *third* return value.
            dropped_turns = included_fulls

            # Skip collapse if turns already collapsed
            if dropped_turns.issubset(collapsed_turns):
                console.print("[green]Dropped turns already collapsed → skipping collapse.[/green]")
                break

            console.print(
                f"[yellow]Dropped turns: {sorted(dropped_turns)} → collapsing…[/yellow]"
            )

            # collapse limit
            collapse_passes += 1
            if collapse_passes > MAX_COLLAPSE_PASSES:
                console.print("[red]Too many collapses → aborting.[/red]")
                break

            # mid-turn protection
            if in_progress and not allow_fulltext_collapse:
                console.print("[yellow]Turn in progress → collapse skipped.[/yellow]")
                break

            # perform collapse
            summarize_turn_fulltext_batches(
                pm, batcher, agent,
                scene_working,
                max_batches=1,
                mode="dynamic"
            )

            # reload state
            scene_working = agent.read_active_scene()
            collapsed_turns = parse_collapsed_turns(scene_working)
            in_progress = has_turn_in_progress(scene_working)

            # stop if nothing remains collapsible
            all_turns = sorted(
                int(n)
                for n in re.findall(r"^#\s*Turn\s+(\d+)", scene_working, flags=re.M)
            )
            remaining_turns = [t for t in all_turns if t not in collapsed_turns]

            if len(remaining_turns) <= min_preserved:
                console.print("[yellow]Cannot collapse further → using current batch.[/yellow]")
                break

            continue  # retry after collapse

        # -------------------------------------------------
        # CASE B — BATCH FITS THRESHOLD → DONE
        # -------------------------------------------------
        total_tokens = system_tokens + agent.count_tokens_string(candidate_text)
        if total_tokens <= threshold_tokens:
            break

        # -------------------------------------------------
        # CASE C — Too large → collapse more turns
        # -------------------------------------------------
        console.print(
            f"[yellow]Dynamic batch too large ({total_tokens}) → collapsing…[/yellow]"
        )

        # find oldest uncollapsed turn(s)
        all_turns = sorted(
            int(n)
            for n in re.findall(r"^#\s*Turn\s+(\d+)", scene_working, flags=re.M)
        )
        next_uncollapsed = [t for t in all_turns if t not in collapsed_turns]

        if not next_uncollapsed:
            console.print("[green]All collapsible turns already collapsed → stop.[/green]")
            break

        collapse_passes += 1
        if collapse_passes > MAX_COLLAPSE_PASSES:
            console.print("[red]Too many collapses → using current batch.[/red]")
            break

        # mid-turn protection
        if in_progress and not allow_fulltext_collapse:
            console.print("[yellow]Turn in progress → collapse skipped.[/yellow]")
            break

        # perform collapse
        summarize_turn_fulltext_batches(
            pm, batcher, agent,
            scene_working,
            max_batches=1,
            mode="dynamic"
        )

        # reload state
        scene_working = agent.read_active_scene()
        collapsed_turns = parse_collapsed_turns(scene_working)
        in_progress = has_turn_in_progress(scene_working)

        # stop if remaining collapsible turns <= min_preserved
        remaining_turns = [t for t in all_turns if t not in collapsed_turns]
        if len(remaining_turns) <= min_preserved:
            console.print("[red]Impossible to reduce context within specified parameters.[/red]")
            console.print("[cyan]Using next-best batch. Type /help perfs for optimization tips.[/cyan]")
            break

    scrunch_check(agent)
    return candidate_text, included_summaries, included_fulls


# ---------------------------------------------------------
# INDIVIDUAL TURN SUMMARIZATION
# ---------------------------------------------------------
def summarize_individual_scene_turns(turn_text, turn_num):
    """
    Summarize a single Turn N block via LLM.
    """
    summary_messages = pm.build_turn_summary_messages(turn_text, turn_num)
    summary = llm_generate_for_summary(summary_messages)
    return summary.strip()


# ---------------------------------------------------------
# REMOVE EMPTY TURNS + GENERAL SUMMARIZATION
# (Now heavily simplified using SceneDocument + utils)
# ---------------------------------------------------------
def summarize_scene_turns(agent, turns_to_keep: int = None):
    """
    Manual turn summarization, keeping last N turns unsummarized.
    """
    scene_path = agent.get_active_scene_path()

    if not scene_path or not scene_path.exists():
        console.print("[red]Scene file not found.[/red]")
        return

    if turns_to_keep is None:
        turns_to_keep = FULL_TURNS_TO_KEEP

    console.print("\n[cyan]Starting turn summarization…[/cyan]\n")

    while True:
        # Clean empty turns
        current_text = scene_path.read_text(encoding="utf-8")
        cleaned = clean_scene_turns(current_text)
        scene_path.write_text(cleaned, encoding="utf-8")

        # Parse with SceneDocument
        doc = SceneDocument(cleaned, agent.count_tokens_string)

        if not doc.turns:
            console.print("[yellow]No turns found to summarize.[/yellow]")
            return

        # Determine turns that need summarizing
        turn_indices = [t["index"] for t in doc.turns]

        if len(turn_indices) <= turns_to_keep:
            to_process = turn_indices
        else:
            to_process = turn_indices[:-turns_to_keep]

        # Find first missing summary
        next_turn = None
        for idx in to_process:
            summary = doc.summary(idx)
            full = doc.full_turn(idx)
            if summary is not None:
                continue
            if not full:
                continue  # empty or invalid turn

            next_turn = idx
            break

        if not next_turn:
            break

        console.print(f"[cyan]→ Summarizing Turn {next_turn}…[/cyan]")

        # Build turn block text
        full_text = doc.full_turn(next_turn)
        block_text = f"# Turn {next_turn}\n{full_text}"

        summary = summarize_individual_scene_turns(block_text, next_turn)

        # Inject summary
        lines = cleaned.splitlines()
        new_lines = []
        inserted = False

        for i, line in enumerate(lines):
            m = TURN_HEADER_RE.match(line)
            if m and int(m.group(1)) == next_turn:
                new_lines.append(f"# Turn {next_turn}")
                new_lines.append("## Summary")
                new_lines.append(summary)
                new_lines.append("")
                new_lines.append("## Full Turn")
                continue

            new_lines.append(line)

        new_text = "\n".join(new_lines)
        scene_path.write_text(new_text, encoding="utf-8")
        cleaned = new_text

        console.print(f"[green]✓ Turn {next_turn} summarized.[/green]\n")

    # Renumber at the end
    final = renumber_turns(scene_path)
    console.print(f"[cyan]Turns renumbered 1 → {final}[/cyan]\n")


# ---------------------------------------------------------
# FULLTEXT SCENE COLLAPSE (refactored but unchanged logic)
# ---------------------------------------------------------
def summarize_turn_fulltext_batches(
    pm,
    batcher,
    agent,
    scene_text=None,
    max_batches=None,
    mode="dynamic"
):
    """
    Collapse full scene into summary batches.
    (Logic cleaned only lightly; mostly intact.)
    """
    console.print("[cyan]Summarizing full scene…[/cyan]")

    if scene_text is None:
        scene_text = agent.read_active_scene()

    doc = SceneDocument(scene_text, agent.count_tokens_string)
    collapsed_turns = doc.collapsed_turns()

    all_turns = [t["index"] for t in doc.turns]
    last_turn = max(all_turns) if all_turns else 0

    # Determine preserved turns
    preserve_turns = set()

    if mode == "dynamic":
        # Keep last FULL turns and MIN summary turns
        for t in range(last_turn - FULL_TURNS_TO_KEEP + 1, last_turn + 1):
            if t > 0:
                preserve_turns.add(t)

        for t in range(last_turn - FULL_TURNS_TO_KEEP - MIN_SUMMARY_TURNS_TO_KEEP + 1,
                       last_turn - FULL_TURNS_TO_KEEP + 1):
            if t > 0:
                preserve_turns.add(t)

    elif mode == "end":
        preserve_turns = collapsed_turns.copy()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    threshold = CONTEXT_THRESHOLD if mode == "dynamic" else SCENE_CONTEXT_THRESHOLD

    summary_batches = batcher.get_tokenwise_summary_batches_excluding(
        scene_text=scene_text,
        system_prompts=[{"role": "system", "content": pm.summary_prompt()}],
        preserve_turns=preserve_turns,
        SCENE_CONTEXT_THRESHOLD=threshold,
        model_token_limit=DEFAULT_MODEL_TOKEN_LIMIT,
    )

    # If nothing to summarize
    any_turns = any(b["turn_indices"] for b in summary_batches)
    if not any_turns:
        console.print("[green]All turns already collapsed.[/green]")
        return scene_text

    if max_batches and len(summary_batches) > max_batches:
        summary_batches = summary_batches[:max_batches]
        console.print(f"[yellow]Limiting to {max_batches} batches…[/yellow]")

    accumulated_summary = ""
    new_collapsed = set(collapsed_turns)

    for idx, batch in enumerate(summary_batches, start=1):
        if not batch["turn_indices"]:
            continue

        console.print(f"\n[bold cyan]Batch {idx}/{len(summary_batches)}[/bold cyan]")
        console.print(f"[magenta]Turns: {batch['turn_indices']}[/magenta]")
        console.print("[yellow]Requesting LLM summary…[/yellow]")

        messages = pm.build_summary_messages(
            scene_text=batch["batch_text"],
            prior_summary_text=batch["prior_summary_text"]
        )

        llm_output = llm_generate_for_summary(messages).strip()
        llm_output = re.sub(r"^#.*summary.*$", "", llm_output, flags=re.I).strip()

        accumulated_summary = join_blocks(accumulated_summary, llm_output)
        new_collapsed.update(batch["turn_indices"])

    # Write updated Scene Summary
    collapsed_str = ",".join(str(n) for n in sorted(new_collapsed))
    description_text = doc.description()

    new_summary_block = (
        f"# Scene Summary (Collapsed: {collapsed_str})\n"
        f"{accumulated_summary}"
    )

    # Insert after description
    updated_scene = scene_text

    # Remove old summary
    updated_scene = re.sub(
        r"(?ms)^#{1,3}\s*Scene Summary.*?(?=^# Turn|\Z)",
        "",
        updated_scene
    ).strip()

    if description_text:
        # Replace description block + insert summary
        updated_scene = re.sub(
            r"(?ms)^# Description.*?(?=^# Turn|\Z)",
            f"# Description\n{description_text}\n\n{new_summary_block}\n\n",
            updated_scene
        )
    else:
        updated_scene = new_summary_block + "\n\n" + updated_scene

    scene_path = agent.get_active_scene_path()
    scene_path.write_text(updated_scene, encoding="utf-8")

    console.print("[bold green]Full scene summary written.[/bold green]")
    return updated_scene


# ---------------------------------------------------------
# TURN RENUMBERING
# ---------------------------------------------------------
def renumber_turns(scene_path: Path):
    """
    Reassign turn numbers sequentially.
    """
    text = scene_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    new_lines = []
    new_num = 1

    for line in lines:
        m = TURN_HEADER_RE.match(line)
        if m:
            new_lines.append(f"# Turn {new_num}")
            new_num += 1
        else:
            new_lines.append(line)

    new_text = "\n".join(new_lines)
    scene_path.write_text(new_text, encoding="utf-8")
    return new_num - 1

# ---------------------------------------------------------
# MIGHTY SCRUNCH - SCRUNCH SMASH !
# ---------------------------------------------------------

def scrunch_check(agent):
    """
    Non-destructive: warn if the Scene Summary exceeds warn_percent of usable context.
    usable context = DEFAULT_MODEL_TOKEN_LIMIT * CONTEXT_THRESHOLD
    """
    warn_percent=0.2
    scene_path = agent.get_active_scene_path()
    if not scene_path or not scene_path.exists():
        return 0

    text = agent.read_active_scene()

    # Detect the Scene Summary block
    match = re.search(
        r"(?ms)^#{1,3}\s*Scene Summary.*?(?=^# Turn|\Z)",
        text
    )

    if not match:
        return 0

    summary_block = match.group(0).strip()
    summary_tokens = agent.count_tokens_string(summary_block)

    # Compute usable token budget
    max_context = DEFAULT_MODEL_TOKEN_LIMIT
    usable_tokens = max_context * CONTEXT_THRESHOLD

    # Warn threshold based on a percentage of usable context
    warn_threshold = usable_tokens * warn_percent

    if summary_tokens > warn_threshold:
        console.print(
            f"[yellow]⚠ Scene Summary is {summary_tokens} tokens "
            f"({summary_tokens / usable_tokens:.1%} of usable context).[/yellow]"
        )
        console.print(
            "[yellow]Use /scrunch to compress it and free room for more turns.[/yellow]"
        )

    return summary_tokens

def scrunch(agent, pm):
    """
    Compress Scene Summary ONLY when explicitly requested.
    Replaces the Scene Summary block with a shorter version.
    """

    scene_path = agent.get_active_scene_path()
    if not scene_path or not scene_path.exists():
        console.print("[red]No active scene to scrunch.[/red]")
        return

    text = agent.read_active_scene()

    # Find scene summary block
    match = re.search(
        r"(?ms)^#{1,3}\s*Scene Summary.*?(?=^# Turn|\Z)",
        text
    )

    if not match:
        console.print("[yellow]No Scene Summary block found.[/yellow]")
        return

    block = match.group(0).strip()
    tokens_before = agent.count_tokens_string(block)

    console.print(f"[cyan]Scrunching Scene Summary ({tokens_before} tokens)…[/cyan]")

    # Summarize the summary
    messages = pm.build_summary_messages(
        scene_text=block,
        prior_summary_text=""
    )
    new_summary_text = llm_generate_for_summary(messages).strip()

    # Preserve Scene Summary header
    first_line = block.splitlines()[0]
    header = first_line if "Scene Summary" in first_line else "# Scene Summary"

    # Rebuild the block
    new_block = f"{header}\n{new_summary_text}\n"

    updated = (
        text[:match.start()] +
        new_block +
        "\n" +
        text[match.end():]
    )

    scene_path.write_text(updated, encoding="utf-8")

    tokens_after = agent.count_tokens_string(new_block)

    console.print(
        f"[green]✔ Scrunch complete — new summary is {tokens_after} tokens.[/green]"
    )

    return tokens_after