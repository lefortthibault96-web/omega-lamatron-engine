import math
import re
from utils import (
    SceneDocument,
    extract_description,
    parse_sections,
    extract_groups,
    get_turn_full_text,
    get_turn_summary_text,
    parse_collapsed_turns,
    join_blocks,
    compute_tokenwise_batches,
    build_batch_text,
)
from config import (
    FULL_TURNS_TO_KEEP,
    MIN_SUMMARY_TURNS_TO_KEEP,
    MAX_SUMMARY_TURNS_TO_KEEP,
)

DEBUG_DYNAMIC_BATCH = True

class BatchManager:
    """
    The BatchManager is responsible for:
      - tokenwise batching of markdown sections
      - full-scene summarization batching
      - dynamic batching for on-the-fly scene collapsing

    This refactored version now uses:
      - SceneDocument for parsing
      - the new utilities in utils.py for consistency
    """

    def __init__(self, agent):
        self.agent = agent

    # ---------------------------------------------------------
    # GENERIC PARSERS (SceneDocument-based)
    # ---------------------------------------------------------
    def parse_generic(self, text: str):
        """
        Wraps new SceneDocument, returning section list.
        """
        doc = SceneDocument(text, self.agent.count_tokens_string)
        return doc.sections

    def extract_groups_from_sections(self, sections, header_regex):
        if isinstance(header_regex, str):
            pattern = re.compile(header_regex)
        else:
            pattern = header_regex

        groups = []
        current_group = None

        for sec in sections:
            header = sec.get("header") or ""
            m = pattern.match(header)

            # If this section starts a new Turn
            if m:
                # Close previous group if any
                if current_group is not None:
                    groups.append(current_group)

                turn_index = int(m.group(1))

                current_group = {
                    "index": turn_index,
                    "sections": []
                }

            # Add sections to current group
            if current_group is not None:
                current_group["sections"].append(sec)

        # Close last group
        if current_group is not None:
            groups.append(current_group)

        return groups

    # ---------------------------------------------------------
    # TOKENWISE BATCH HELPERS (unchanged logic)
    # ---------------------------------------------------------
    def get_tokenwise_batches(self, blocks, system_prompts, threshold):
        """
        Computes tokenwise batches while respecting system prompt tokens.
        """
        system_tokens, _ = self.agent.count_tokens(
            messages=system_prompts,
            return_breakdown=True
        )
        return compute_tokenwise_batches(blocks, system_tokens, threshold)

    def build_tokenwise_text(self, batches, label_prefix="Block"):
        """
        Convert tokenwise batch data to readable markdown.
        """
        return build_batch_text(batches, label_prefix)

    # ---------------------------------------------------------
    # FULL SCENE SUMMARY BATCHING
    # ---------------------------------------------------------
    def get_tokenwise_summary_batches(
        self,
        scene_text: str,
        system_prompts: list[dict],
        SCENE_CONTEXT_THRESHOLD: float,
        model_token_limit: int
    ):
        """
        Distribute a full scene into several summary batches.
        Uses SceneDocument for all parsing.
        """

        doc = SceneDocument(scene_text, self.agent.count_tokens_string)
        turns = doc.turns
        description_text = doc.description()

        # Token accounting
        system_tokens = self.agent.count_tokens(system_prompts)
        desc_tokens = self.agent.count_tokens_string(description_text) if description_text else 0

        turn_token_counts = []
        total_tokens = system_tokens + desc_tokens

        # Calculate each turn’s tokens
        for turn in turns:
            full = get_turn_full_text(turn)
            turn_block = f"# Turn {turn['index']}\n{full}\n"
            tks = self.agent.count_tokens_string(turn_block)

            turn_token_counts.append(tks)
            total_tokens += tks

        max_batch_tokens = int(model_token_limit * SCENE_CONTEXT_THRESHOLD)
        num_batches = max(1, math.ceil(total_tokens / max_batch_tokens))

        batches = []
        prior_summary_text = ""
        turn_idx = 0
        turns_per_batch = math.ceil(len(turns) / num_batches)

        # Build batches
        for batch_num in range(num_batches):
            lines = []

            # Include description only in first batch
            if batch_num == 0 and description_text:
                lines.append(f"# Description\n{description_text}")

            if prior_summary_text:
                lines.append(prior_summary_text + "\n")

            batch_turn_indices = []

            for _ in range(turns_per_batch):
                if turn_idx >= len(turns):
                    break

                turn = turns[turn_idx]
                full = get_turn_full_text(turn)
                turn_block = f"# Turn {turn['index']}\n{full}\n"

                lines.append(turn_block)
                batch_turn_indices.append(turn["index"])

                turn_idx += 1

            batch_text = "\n".join(lines).strip()

            batches.append({
                "batch_text": batch_text,
                "prior_summary_text": prior_summary_text,
                "turn_indices": batch_turn_indices,
            })

        return batches

    # ---------------------------------------------------------
    # SUMMARY BATCHING WITH EXCLUDED TURNS
    # ---------------------------------------------------------
    def get_tokenwise_summary_batches_excluding(
        self,
        scene_text: str,
        system_prompts: list[dict],
        preserve_turns: set[int],
        SCENE_CONTEXT_THRESHOLD: float,
        model_token_limit: int
    ):
        """
        Same as get_tokenwise_summary_batches, but excludes:
          - preserved-turns
          - already-collapsed turns
        """

        doc = SceneDocument(scene_text, self.agent.count_tokens_string)
        collapsed = doc.collapsed_turns()

        # Exclude preserved + collapsed
        exclude = preserve_turns | collapsed
        turns = [t for t in doc.turns if t["index"] not in exclude]

        description_text = doc.description()

        # Token accounting
        system_tokens = self.agent.count_tokens(system_prompts)
        desc_tokens = self.agent.count_tokens_string(description_text) if description_text else 0

        turn_token_counts = []
        total_tokens = system_tokens + desc_tokens

        # Compute tokens per remaining turn
        for turn in turns:
            full = get_turn_full_text(turn)
            turn_block = f"# Turn {turn['index']}\n{full}\n"
            tks = self.agent.count_tokens_string(turn_block)

            turn_token_counts.append(tks)
            total_tokens += tks

        max_batch_tokens = int(model_token_limit * SCENE_CONTEXT_THRESHOLD)
        num_batches = max(1, math.ceil(total_tokens / max_batch_tokens))

        batches = []
        prior_summary_text = ""
        turn_idx = 0
        turns_per_batch = math.ceil(len(turns) / num_batches) if turns else 0

        # Build batches
        for batch_num in range(num_batches):
            lines = []

            if batch_num == 0 and description_text:
                lines.append(f"# Description\n{description_text}")

            if prior_summary_text:
                lines.append(prior_summary_text + "\n")

            batch_turn_indices = []

            for _ in range(turns_per_batch):
                if turn_idx >= len(turns):
                    break

                turn = turns[turn_idx]
                full = get_turn_full_text(turn)
                turn_block = f"# Turn {turn['index']}\n{full}\n"

                lines.append(turn_block)
                batch_turn_indices.append(turn["index"])

                turn_idx += 1

            batch_text = "\n".join(lines).strip()

            batches.append({
                "batch_text": batch_text,
                "prior_summary_text": prior_summary_text,
                "turn_indices": batch_turn_indices,
            })

        return batches

    # ---------------------------------------------------------
    # DYNAMIC SUMMARIZATION BATCH BUILDER
    # (Heavily simplified using SceneDocument)
    # ---------------------------------------------------------

    def build_dynamic_summarization_batch(
        self,
        scene_text: str,
        threshold_tokens: int,
        ignored_turns=None,
        system_prompts=None,
        default_prompt_builder=None,
    ):
        # ---------------------------------------------------------
        # CONFIG FLAG
        # ---------------------------------------------------------
        DEBUG_DYNAMIC_BATCH = True

        if ignored_turns is None:
            ignored_turns = set()

        # ---------------------------------------------------------
        # SYSTEM TOKENS
        # ---------------------------------------------------------
        if system_prompts is None:
            if default_prompt_builder is None:
                raise ValueError("Must provide default_prompt_builder")
            system_prompts = default_prompt_builder(scene_text="")

        system_tokens = sum(self.agent.count_tokens_string(p["content"]) for p in system_prompts)
        available_tokens = threshold_tokens - system_tokens
        if available_tokens <= 0:
            return "", [], []

        # ---------------------------------------------------------
        # PARSE SCENE
        # ---------------------------------------------------------
        doc = SceneDocument(scene_text, self.agent.count_tokens_string)
        sections = doc.sections
        turns = doc.turns
        collapsed = doc.collapsed_turns() | ignored_turns

        # ---------------------------------------------------------
        # MANDATORY BLOCKS
        # ---------------------------------------------------------
        description_text = doc.description() or ""
        description_block = f"# Description\n{description_text}" if description_text else ""

        scene_summary_text = ""
        for sec in sections:
            if (sec["header"] or "").lower().startswith("scene summary"):
                scene_summary_text = f"# Scene Summary\n{sec['text']}"
                break

        # ---------------------------------------------------------
        # ONGOING TURN (structural rule)
        # ---------------------------------------------------------
        inprog_text = ""
        inprog_idx = None

        if turns:
            last = turns[-1]
            idx = last["index"]

            # Default: assume NOT in-progress
            inprog_idx = None

            has_summary = any((s["header"] or "").lower() == "summary" for s in last["sections"])
            has_full    = any((s["header"] or "").lower() == "full turn" for s in last["sections"])

            # Collect body text
            body_chunks = []
            for sec in last["sections"]:
                h = (sec["header"] or "").lower()
                if h in ("summary", "full turn"):
                    continue
                if sec["text"].strip():
                    body_chunks.append(sec["text"])

            # IN-PROGRESS detection is ONLY here
            if body_chunks and not has_summary and not has_full:
                body = "\n\n".join(body_chunks)
                inprog_text = f"# Turn {idx} (In Progress)\n{body}"
                inprog_idx = idx     # ✔ Only mark in-progress when it actually is

        # ---------------------------------------------------------
        # TOKEN COST OF MANDATORY BLOCKS
        # ---------------------------------------------------------
        mand_text = "\n\n".join(filter(None, [
            description_block,
            scene_summary_text,
            inprog_text
        ]))

        mand_cost = self.agent.count_tokens_string(mand_text)
        remaining = max(0, available_tokens - mand_cost)

        # ---------------------------------------------------------
        # BUILD CANDIDATE TURNS (excluding collapsed + ongoing)
        # ---------------------------------------------------------
        turn_summaries = {}
        turn_fulls = {}
        candidate_turns = []

        for group in turns:
            idx = group["index"]
            if idx == inprog_idx:
                continue
            if idx in collapsed:
                continue

            summary = get_turn_summary_text(group)
            full = get_turn_full_text(group)

            if summary:
                turn_summaries[idx] = summary
            if full:
                turn_fulls[idx] = full

            candidate_turns.append(idx)

        candidate_turns.sort()

        # ---------------------------------------------------------
        # REQUIRED FULLS / SUMMARIES
        # ---------------------------------------------------------
        # Required full turns = newest N fulls
        full_candidates = [i for i in candidate_turns if i in turn_fulls]
        if FULL_TURNS_TO_KEEP > 0:
            required_full = sorted(full_candidates)[-FULL_TURNS_TO_KEEP:]
        else:
            required_full = []
        required_full = sorted(required_full)

        # Required summaries = oldest M summaries
        summary_candidates = [i for i in candidate_turns
                            if i not in required_full and i in turn_summaries]

        if MIN_SUMMARY_TURNS_TO_KEEP > 0:
            required_summaries = summary_candidates[:MIN_SUMMARY_TURNS_TO_KEEP]
        else:
            required_summaries = []
        required_summaries = sorted(required_summaries)

        # ---------------------------------------------------------
        # BASELINE CONFIG: ALL SUMMARIES (minus required full)
        # ---------------------------------------------------------
        current_fulls = set(required_full)
        current_summaries = set(
            idx for idx in candidate_turns
            if idx not in current_fulls and idx in turn_summaries
        )
        current_summaries |= set(required_summaries)

        # Prepare baseline blocks
        def build_blocks(summary_set, full_set):
            blocks = []
            for i in sorted(summary_set):
                blocks.append(f"## Turn {i} Summary\n{turn_summaries[i]}")
            for i in sorted(full_set):
                blocks.append(f"# Turn {i}\n{turn_fulls[i]}")
            return blocks

        baseline_blocks = build_blocks(current_summaries, current_fulls)
        baseline_text = mand_text + "\n\n" + "\n\n".join(baseline_blocks)
        baseline_cost = self.agent.count_tokens_string(baseline_text)

        # ---------------------------------------------------------
        # IF BASELINE TOO LARGE → RETURN BASELINE AS-IS
        # ---------------------------------------------------------
        if baseline_cost > threshold_tokens:
            if DEBUG_DYNAMIC_BATCH:
                print("\n===== DYNAMIC BATCH DEBUG (BASELINE ONLY — TOO LARGE) =====")
                print(f"Summaries: {sorted(current_summaries)}")
                print(f"Fulls    : {sorted(current_fulls)}")
                print(f"Usage    : {baseline_cost} / {threshold_tokens}")
                print("===========================================================\n")

            # final assembly
            final_parts = [p for p in [
                description_block,
                scene_summary_text,
                "\n\n".join(baseline_blocks),
                inprog_text
            ] if p]

            final_text = "\n\n".join(final_parts).strip()

            # dropped-turn detection
            original_summary = set(turn_summaries.keys())
            included_summary = set(current_summaries)
            included_full = set(current_fulls)
            dropped = original_summary - included_summary - included_full

            if dropped:
                return None, None, dropped

            return final_text, [(i, turn_summaries[i]) for i in sorted(current_summaries)], [(i, turn_fulls[i]) for i in sorted(current_fulls)]

        # ---------------------------------------------------------
        # BASELINE FITS → TRY FULL-TURN UPGRADES (NEWEST→OLDEST)
        # ---------------------------------------------------------
        best_cost = baseline_cost
        best_fulls = sorted(current_fulls)
        best_summaries = sorted(current_summaries)

        for idx in reversed(candidate_turns):
            # Upgrade summary→full if possible
            if idx in turn_fulls:
                current_fulls.add(idx)
            if idx in current_summaries:
                current_summaries.remove(idx)

            blocks = build_blocks(current_summaries, current_fulls)
            cfg_text = mand_text + "\n\n" + "\n\n".join(blocks)
            cfg_cost = self.agent.count_tokens_string(cfg_text)

            if cfg_cost <= threshold_tokens and cfg_cost > best_cost:
                best_cost = cfg_cost
                best_fulls = sorted(current_fulls)
                best_summaries = sorted(current_summaries)
            else:
                # Stop upgrading once over budget
                break

        # ---------------------------------------------------------
        # FINAL ASSEMBLY (AFTER UPGRADES)
        # ---------------------------------------------------------
        final_blocks = build_blocks(best_summaries, best_fulls)

        final_parts = [p for p in [
            description_block,
            scene_summary_text,
            "\n\n".join(final_blocks),
            inprog_text
        ] if p]

        final_text = "\n\n".join(final_parts).strip()

        if DEBUG_DYNAMIC_BATCH:
            print("\n===== DYNAMIC BATCH DEBUG (FINAL CONFIG) =====")
            print(f"Summaries: {best_summaries}")
            print(f"Fulls    : {best_fulls}")
            print(f"Usage    : {best_cost} / {threshold_tokens}")
            print("==============================================\n")

            # ---------------------------------------------------------
            # ENFORCE MAX SUMMARY KEEP
            # ---------------------------------------------------------
        if MAX_SUMMARY_TURNS_TO_KEEP is not None:
                if len(best_summaries) > MAX_SUMMARY_TURNS_TO_KEEP:
                    # oldest ones get dropped
                    num_to_drop = len(best_summaries) - MAX_SUMMARY_TURNS_TO_KEEP
                    dropped = set(best_summaries[:num_to_drop])
                    return None, None, dropped

        # dropped-turn detection
        original_summary = set(turn_summaries.keys())
        included_summary = set(best_summaries)
        included_full = set(best_fulls)
        dropped = original_summary - included_summary - included_full

        if dropped:
            return None, None, dropped

        return (
            final_text,
            [(i, turn_summaries[i]) for i in best_summaries],
            [(i, turn_fulls[i]) for i in best_fulls]
        )

