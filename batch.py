import math
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

    def extract_groups_from_sections(self, sections, header_regex=r"^Turn\s+(\d+)", ignore_case=True):
        """
        Groups are extracted using utils.extract_groups indirectly
        via SceneDocument.
        """
        text = "\n".join(sec["text"] for sec in sections)
        doc = SceneDocument(text, self.agent.count_tokens_string)
        return doc.turns

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
        system_prompts: list[dict] = None,
        default_prompt_builder=None,
    ):

        if ignored_turns is None:
            ignored_turns = set()

        # Step 0…
        if system_prompts is None:
            if default_prompt_builder is None:
                raise ValueError("Must provide default_prompt_builder if system_prompts is None")
            system_prompts = default_prompt_builder(scene_text="")

        system_tokens = sum(self.agent.count_tokens_string(p["content"]) for p in system_prompts)
        available_tokens = threshold_tokens - system_tokens
        if available_tokens <= 0:
            raise ValueError(
                f"System content ({system_tokens} tokens) exceeds total threshold ({threshold_tokens})"
            )

        # Step 2…  (use utils)
        sections = parse_sections(scene_text, self.agent.count_tokens_string)
        turns = extract_groups(sections, header_regex=r"^Turn\s+(\d+)")

        # Step 3… (use utils.extract_description)
        raw_desc = extract_description(sections)
        description_text = f"# Description\n{raw_desc}" if raw_desc else ""

        # Extract scene summary (utils has no helper for this header, so keep your logic)
        scene_summary_text = ""
        for sec in sections:
            header = (sec["header"] or "").lower()
            if header.startswith("scene summary"):
                scene_summary_text = f"# Scene Summary\n{sec['text']}"

        # Parse collapsed metadata from the Scene Summary header
        collapsed_turns = parse_collapsed_turns(scene_text)
        collapsed = ignored_turns | collapsed_turns

        # Step 4…
        uncollapsed_summaries = []
        full_turn_texts = {}
        finished_indices = []
        turn_in_progress_text = None

        for group in turns:
            idx = group["index"]

                # Utils helper for summary
            summary = get_turn_summary_text(group)

            # Detect real "Full Turn" section (## Full Turn)
            has_full = any(
                (sec.get("header") or "").lower() == "full turn"
                and sec.get("level") == 2
                for sec in group["sections"]
            )

            # Only get full text if "Full Turn" section truly exists
            full = get_turn_full_text(group) if has_full else None

            # Detect "(In Progress)"
            first_header = (group["sections"][0]["header"] or "").lower() if group["sections"] else ""
            is_in_progress = "in progress" in first_header

            # Register summaries
            if summary and idx not in collapsed:
                uncollapsed_summaries.append((idx, summary))

            # Register full turns
            if full and idx not in collapsed:
                full_turn_texts[idx] = full
                finished_indices.append(idx)

            # Collect in-progress content (non-summary + non-full)
            if is_in_progress:
                chunks = []
                for sec in group["sections"]:
                    h = (sec["header"] or "").lower()
                    if h in ("summary", "full turn"):
                        continue
                    if sec["text"].strip():
                        chunks.append(sec["text"])

                merged = join_blocks(*chunks)
                turn_in_progress_text = f"# Turn {idx} (In Progress)\n{merged}"

        # Step 5…
        included_summaries = list(uncollapsed_summaries)
        included_fulls = []
        min_full_turns = FULL_TURNS_TO_KEEP

        for idx in reversed(finished_indices):
            if idx in collapsed:
                continue

            parts = []

            if description_text:
                parts.append(description_text)
            if scene_summary_text:
                parts.append(scene_summary_text)

            # Summaries except this turn
            for s_idx, s_txt in included_summaries:
                if s_idx not in [f_idx for f_idx, _ in included_fulls] and s_idx != idx:
                    parts.append(f"## Turn {s_idx} Summary\n{s_txt}")

            # Already included full turns
            for f_idx, f_txt in included_fulls:
                parts.append(f"# Turn {f_idx}\n{f_txt}")

            # Candidate full turn
            parts.append(f"# Turn {idx}\n{full_turn_texts[idx]}")

            if turn_in_progress_text:
                parts.append(turn_in_progress_text)

            test_text = join_blocks(*parts)
            if self.agent.count_tokens_string(test_text) <= available_tokens:
                included_fulls.append((idx, full_turn_texts[idx]))
            else:
                break

        # Minimum full-turn guarantee
        if len(included_fulls) < min_full_turns and finished_indices:
            for newest in reversed(finished_indices):
                if newest not in collapsed:
                    included_fulls = [(newest, full_turn_texts[newest])]
                    break

        # Remove summaries that have full turns
        full_indices = {idx for idx, _ in included_fulls}
        included_summaries = [
            (idx, txt) for (idx, txt) in included_summaries
            if idx not in full_indices and idx not in collapsed
        ]
        included_summaries.sort(key=lambda x: x[0])

        # Enforce MAX summary limit
        if MAX_SUMMARY_TURNS_TO_KEEP is not None and len(included_summaries) > MAX_SUMMARY_TURNS_TO_KEEP:
            included_summaries = included_summaries[-MAX_SUMMARY_TURNS_TO_KEEP:]

        # Enforce MIN summary limit
        if MIN_SUMMARY_TURNS_TO_KEEP is not None and len(included_summaries) < MIN_SUMMARY_TURNS_TO_KEEP:
            shortage = MIN_SUMMARY_TURNS_TO_KEEP - len(included_summaries)

            # Only consider candidates that are:
            #  - not already included as full
            #  - not ignored/collapsed
            candidates = [
                (idx, txt)
                for (idx, txt) in sorted(uncollapsed_summaries, key=lambda x: x[0])
                if idx not in full_indices and idx not in collapsed
            ]

            needed = []
            for s in reversed(candidates):
                if s not in included_summaries:
                    needed.append(s)
                if len(needed) >= shortage:
                    break

            included_summaries.extend(reversed(needed))

        included_summaries.sort(key=lambda x: x[0])

        # SPECIAL CASE: total collapse
        if not uncollapsed_summaries and not full_turn_texts:
            if MIN_SUMMARY_TURNS_TO_KEEP > 0:
                summary_text = scene_summary_text.replace("# Scene Summary", "").strip()
                words = summary_text.split()

                if not words:
                    final_parts = []
                    if description_text:
                        final_parts.append(description_text)
                    final_parts.append(f"# Scene Summary\n{summary_text}")
                    final_text = join_blocks(*final_parts)
                    return final_text, [], []

                chunk_size = max(1, len(words) // MIN_SUMMARY_TURNS_TO_KEEP)
                synthetic_summaries = []

                for i in range(MIN_SUMMARY_TURNS_TO_KEEP):
                    chunk = words[i * chunk_size : (i + 1) * chunk_size]
                    if not chunk:
                        break
                    synthetic_summaries.append((i + 1, " ".join(chunk)))

                final_parts = []
                if description_text:
                    final_parts.append(description_text)
                final_parts.append(f"# Scene Summary\n{summary_text}")

                for idx, txt in synthetic_summaries:
                    final_parts.append(f"## Turn {idx} Summary\n{txt}")

                if turn_in_progress_text:
                    final_parts.append(turn_in_progress_text)

                final_text = join_blocks(*final_parts)
                return final_text, synthetic_summaries, []

        # Detect dropped turns
        original_summary_indices = {idx for idx, _ in uncollapsed_summaries}
        included_summary_indices = {idx for idx, _ in included_summaries}
        included_full_indices = {idx for idx, _ in included_fulls}

        dropped_turns = original_summary_indices - included_summary_indices - included_full_indices
        if dropped_turns:
            return None, None, dropped_turns

        # Step 6: final assembly
        final_parts = []

        if description_text:
            final_parts.append(description_text)
        if scene_summary_text:
            final_parts.append(scene_summary_text)

        for s_idx, s_txt in included_summaries:
            if s_idx not in full_indices:
                final_parts.append(f"## Turn {s_idx} Summary\n{s_txt}")

        for f_idx, f_txt in included_fulls:
            final_parts.append(f"# Turn {f_idx}\n{f_txt}")

        if turn_in_progress_text:
            final_parts.append(turn_in_progress_text)

        final_text = join_blocks(*final_parts)
        return final_text, included_summaries, included_fulls