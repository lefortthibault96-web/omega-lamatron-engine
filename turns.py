import re
from config import AUTO_SUMMARIZE, CONTEXT_THRESHOLD, prompts_dir
from rich.console import Console
from pathlib import Path
from config import TURNS_TO_KEEP
from utils import DEFAULT_MODEL_TOKEN_LIMIT

console = Console()

def ensure_current_turn(scene_path):
    """
    Ensures the scene has a # Description section
    and that the last turn header is present at the end.
    Returns the current turn number (last turn found).
    Does NOT create a new turn — that is done in advance_turn().
    """
    if not scene_path or not scene_path.exists():
        console.print("[red]No active scene file found.[/red]")
        return 0  # fallback

    lines = scene_path.read_text(encoding="utf-8").splitlines()
    turn_pattern = re.compile(r"^#\s*Turn\s+(\d+)", re.I)

    # --- Ensure Description exists ---
    has_description = any(line.strip().lower().startswith("# description") for line in lines)
    first_turn_index = next((i for i, l in enumerate(lines) if turn_pattern.match(l)), len(lines))

    if not has_description:
        console.print("[cyan]No # Description section found — creating one…[/cyan]")
        description_content = lines[:first_turn_index]
        remaining = lines[first_turn_index:]
        lines = ["# Description", ""] + description_content + [""] + remaining
    
    # --- Ensure at least one turn exists ---
    turn_numbers = [int(m.group(1)) for l in lines if (m := turn_pattern.match(l))]
    if not turn_numbers:
        console.print("[cyan]No turns found — creating # Turn 1…[/cyan]")

        # Insert Turn 1 after description block
        desc_index = next(
            (i for i, line in enumerate(lines) if line.strip().lower().startswith("# description")),
            -1
        )
        insert_index = desc_index + 1
        while insert_index < len(lines) and not lines[insert_index].strip().startswith("#"):
            insert_index += 1

        lines = lines[:insert_index] + ["", "# Turn 1", ""] + lines[insert_index:]

    # --- Determine last turn ---
    turn_numbers = [int(turn_pattern.match(l).group(1)) for l in lines if turn_pattern.match(l)]
    if turn_numbers:
        last_turn = max(turn_numbers)
    else:
        last_turn = 0  # no turn exists yet

    # --- Save back ---
    scene_path.write_text("\n".join(lines), encoding="utf-8")

    return last_turn

def advance_turn(scene_path, agent, current_turn=None):
    if not scene_path or not scene_path.exists():
        console.print("[red]No active scene file found.[/red]")
        return

    lines = scene_path.read_text(encoding="utf-8").splitlines()
    turn_pattern = re.compile(r"^#\s*Turn\s+(\d+)", re.I)

    # Determine last turn if current_turn not provided
    turn_numbers = [int(turn_pattern.match(l).group(1)) for l in lines if turn_pattern.match(l)]
    last_turn = max(turn_numbers) if turn_numbers else 0

    if current_turn is None:
        current_turn = last_turn

    new_turn = last_turn + 1

    # --- Auto-summary based on token usage ---
    if AUTO_SUMMARIZE and hasattr(agent, "_last_token_usage") and DEFAULT_MODEL_TOKEN_LIMIT:
        usage_ratio = agent._last_token_usage / DEFAULT_MODEL_TOKEN_LIMIT
        if usage_ratio >= CONTEXT_THRESHOLD:
            console.print(f"[yellow]Token usage {usage_ratio*100:.1f}% — auto-summarizing previous turns[/yellow]")
            summarize_scene_turns(scene_path, agent)
            # Refresh lines after summarization
            lines = scene_path.read_text(encoding="utf-8").splitlines()

    # --- Append new turn ---
    console.print(f"[cyan]Advancing to Turn {new_turn}…[/cyan]")
    lines.append(f"# Turn {new_turn}")
    scene_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[bold green]Turn {new_turn} created.[/bold green]")

    return new_turn



def summarize_turn(turn_text, agent, turn_num):
    """
    Sends a single turn text to the LLM for summarization.
    Returns the summary string.
    """
    from Prompt_Manager2000 import PromptManager
    pm = PromptManager(prompts_dir)
    summary_messages = pm.build_turn_summary_messages(turn_text, turn_num)
    summary = agent.chat(summary_messages).strip()
    return summary


turn_header_pattern = re.compile(
    r"^#{1,3}\s*Turn[: ]+\s*(\d+)",
    re.IGNORECASE
)



# =====================================================================
#   RE-NUMBER ALL TURNS AFTER SUMMARIZATION + REMOVALS
# =====================================================================
def renumber_turns(scene_path: Path):
    lines = scene_path.read_text(encoding="utf-8").splitlines()

    new_lines = []
    new_turn_num = 1

    for line in lines:
        m = turn_header_pattern.match(line)
        if m:
            # Replace turn header with new consecutive number
            new_lines.append(f"# Turn {new_turn_num}")
            new_turn_num += 1
        else:
            new_lines.append(line)

    scene_path.write_text("\n".join(new_lines), encoding="utf-8")
    return new_turn_num - 1  # number of turns



# =====================================================================
#   REMOVE EMPTY TURNS
# =====================================================================
def remove_empty_turns(lines, console):
    turn_positions = [(i, int(m.group(1))) for i, line in enumerate(lines) if (m := turn_header_pattern.match(line))]
    to_delete_ranges = []

    for idx, (line_idx, turn_num) in enumerate(turn_positions):
        next_idx = turn_positions[idx + 1][0] if idx + 1 < len(turn_positions) else len(lines)
        block = lines[line_idx:next_idx]

        # Remove all headers
        content_lines = [
            l for l in block
            if not l.strip().lower().startswith("# turn")
            and not l.strip().lower().startswith("## summary")
            and not l.strip().lower().startswith("## full")
        ]

        # If block contains no actual content → mark for removal
        if all(not l.strip() for l in content_lines):
            console.print(f"[yellow]Removed empty Turn {turn_num}[/yellow]")
            to_delete_ranges.append((line_idx, next_idx))

    # Apply deletions in reverse order (so indexing stays valid)
    for start, end in reversed(to_delete_ranges):
        del lines[start:end]

    return lines



# =====================================================================
#   MAIN SUMMARIZATION FUNCTION
# =====================================================================
def summarize_scene_turns(scene_path: Path, agent, turns_to_keep: int = None):

    console = Console()
    summary_marker = "## Summary"

    if not scene_path or not scene_path.exists():
        console.print("[red]Scene file not found.[/red]")
        return

    if turns_to_keep is None:
        turns_to_keep = TURNS_TO_KEEP

    console.print("\n[cyan]Starting turn summarization…[/cyan]\n")

    while True:
        lines = scene_path.read_text(encoding="utf-8").splitlines()

        # ------------------------
        # Remove empty turns first
        # ------------------------
        lines = remove_empty_turns(lines, console)
        scene_path.write_text("\n".join(lines), encoding="utf-8")

        # Re-scan after removal
        lines = scene_path.read_text(encoding="utf-8").splitlines()
        turn_positions = [(i, int(m.group(1))) for i, line in enumerate(lines) if (m := turn_header_pattern.match(line))]

        if not turn_positions:
            console.print("[yellow]No turns found to summarize.[/yellow]")
            return

        # Determine which turns to summarize (skip last N)
        if turns_to_keep == 0 or len(turn_positions) <= turns_to_keep:
            turns_to_summarize = turn_positions
        else:
            turns_to_summarize = turn_positions[:-turns_to_keep]

        # Find the first turn without a Summary section
        next_turn = None
        for idx, (line_idx, turn_num) in enumerate(turns_to_summarize):
            next_idx = turn_positions[idx + 1][0] if idx + 1 < len(turn_positions) else len(lines)
            block = lines[line_idx:next_idx]
            block_text = "\n".join(block)

            if summary_marker in block_text:
                continue  # already summarized

            # Skip empty ones (already removed above, but double safety)
            non_header = [
                l for l in block
                if not l.strip().lower().startswith("# turn")
                and not l.strip().lower().startswith("## summary")
                and not l.strip().lower().startswith("## full")
            ]
            if all(not l.strip() for l in non_header):
                console.print(f"[yellow]Skipped empty Turn {turn_num}[/yellow]")
                continue

            next_turn = (line_idx, turn_num, next_idx, block)
            break

        if not next_turn:
            break  # no more to summarize

        line_idx, turn_num, next_idx, block = next_turn

        console.print(f"[cyan]→ Turn {turn_num}: generating summary…[/cyan]")

        summary = summarize_turn("\n".join(block), agent, turn_num).strip()

        new_block = [
            f"# Turn {turn_num}",
            "## Summary",
            summary,
            "",
            "## Full Turn",
        ] + block[1:]

        lines[line_idx:next_idx] = new_block
        scene_path.write_text("\n".join(lines), encoding="utf-8")

        console.print(f"[green]✓ Turn {turn_num} summarized.[/green]\n")

    console.print("[bold green]All missing turn summaries completed.[/bold green]")

    # ----------------------------------------------------
    # FINAL STEP → Renumber all turns to be consecutive
    # ----------------------------------------------------
    final_count = renumber_turns(scene_path)
    console.print(f"[cyan]Turns renumbered 1 → {final_count}[/cyan]\n")

def parse_scene_generic(self, scene_text: str) -> list[dict]:
    """
    Parse scene into hierarchical blocks based on headers.
    Returns a list of sections:
      - level: header level (1=#, 2=##, etc.)
      - header: text of the header
      - lines: lines under this header
      - text: full text of this block
      - tokens: token count
    """
    import re

    lines = scene_text.splitlines()
    sections = []
    current_block = None
    header_pattern = re.compile(r"^(#+)\s+(.*)$")  # detects #, ##, ### ...

    for line in lines:
        match = header_pattern.match(line.strip())
        if match:
            # append previous block
            if current_block:
                current_block["text"] = "\n".join(current_block["lines"])
                current_block["tokens"] = self.count_tokens_string(current_block["text"])
                sections.append(current_block)

            level = len(match.group(1))
            header = match.group(2)
            current_block = {"level": level, "header": header, "lines": []}
        else:
            if current_block:
                current_block["lines"].append(line.strip())
            else:
                # content outside any header
                current_block = {"level": 0, "header": None, "lines": [line.strip()]}

    # append last block
    if current_block:
        current_block["text"] = "\n".join(current_block["lines"])
        current_block["tokens"] = self.count_tokens_string(current_block["text"])
        sections.append(current_block)

    return sections

