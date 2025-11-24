import re
import math

class BatchManager:
    def __init__(self, agent):
        self.agent = agent

    # -----------------------------
    # GENERIC PARSING
    # -----------------------------
    def parse_generic(self, text: str) -> list[dict]:
        """
        Parse text into hierarchical sections:
          - level: header level (0 = no header)
          - header: header text or None
          - lines: raw lines
          - text: joined text content
          - tokens: token count
          - id: sequential index
        """
        lines = text.splitlines()
        sections = []
        current_block = None
        header_pattern = re.compile(r"^(#+)\s+(.*)$")

        for line in lines:
            stripped = line.strip()
            match = header_pattern.match(stripped)

            if match:
                # finalize previous block
                if current_block:
                    current_block["text"] = "\n".join(current_block["lines"])
                    current_block["tokens"] = self.agent.count_tokens_string(current_block["text"])
                    sections.append(current_block)

                level = len(match.group(1))
                header = match.group(2)
                current_block = {
                    "level": level,
                    "header": header,
                    "lines": []
                }
            else:
                if current_block:
                    current_block["lines"].append(stripped)
                else:
                    current_block = {
                        "level": 0,
                        "header": None,
                        "lines": [stripped]
                    }

        # append last block
        if current_block:
            current_block["text"] = "\n".join(current_block["lines"])
            current_block["tokens"] = self.agent.count_tokens_string(current_block["text"])
            sections.append(current_block)

        # assign sequential ids
        for i, sec in enumerate(sections):
            sec["id"] = i

        return sections

    # -----------------------------
    # GENERIC GROUP/Turn EXTRACTION
    # -----------------------------
    def extract_groups_from_sections(self, sections: list[dict], header_regex=r"^Turn\s+(\d+)", ignore_case=True) -> list[dict]:
        """
        Extract groups/turns from sections based on header regex.
        Returns list of groups with:
          - index: group number (from regex)
          - sections: list of sections belonging to that group
        """
        flags = re.IGNORECASE if ignore_case else 0
        group_re = re.compile(header_regex, flags)

        groups = []
        current_group = None

        for sec in sections:
            header = sec["header"]
            m = group_re.match(header) if header else None

            if m:
                # start new group
                if current_group:
                    groups.append(current_group)
                group_index = int(m.group(1))
                current_group = {"index": group_index, "sections": []}

            if current_group:
                current_group["sections"].append(sec)

        if current_group:
            groups.append(current_group)

        return groups

    # -----------------------------
    # GENERIC TOKENWISE BATCHING
    # -----------------------------
    def get_tokenwise_batches(self, blocks: list[dict], system_prompts: list[dict], threshold: int) -> list[dict]:
        """
        Produce batches of blocks so that system_tokens + batch_tokens <= threshold.
        Returns list of batches:
          - indices: list of block ids
          - text: joined text of the batch
          - tokens: token count of the batch
        """
        # 1) Count system tokens
        system_tokens, _ = self.agent.count_tokens(messages=system_prompts, return_breakdown=True)
        allowed_tokens = threshold - system_tokens
        if allowed_tokens <= 0:
            raise ValueError(f"System tokens ({system_tokens}) exceed threshold ({threshold})")

        batches = []
        current_blocks = []
        current_tokens = 0

        for block in blocks:
            b_tokens = block["tokens"]
            if current_blocks and (current_tokens + b_tokens > allowed_tokens):
                batches.append({
                    "indices": [b["id"] for b in current_blocks],
                    "text": "\n\n".join(b["text"] for b in current_blocks),
                    "tokens": current_tokens
                })
                current_blocks = []
                current_tokens = 0

            current_blocks.append(block)
            current_tokens += b_tokens

        if current_blocks:
            batches.append({
                "indices": [b["id"] for b in current_blocks],
                "text": "\n\n".join(b["text"] for b in current_blocks),
                "tokens": current_tokens
            })

        return batches

    # -----------------------------
    # GENERIC BUILD TEXT FROM BATCHES
    # -----------------------------
    def build_tokenwise_text(self, batches: list[dict], label_prefix="Block") -> str:
        """
        Build a collapsed text from batches.
        """
        output_lines = []

        for batch in batches:
            indices = batch["indices"]
            start_i = indices[0]
            end_i = indices[-1]

            if start_i == end_i:
                output_lines.append(f"# {label_prefix} {start_i}")
            else:
                output_lines.append(f"# {label_prefix}s {start_i}–{end_i}")

            output_lines.append(batch["text"])
            output_lines.append("")

        return "\n".join(output_lines).strip()

    # -----------------------------
    # SCENE-SPECIFIC WRAPPERS
    # -----------------------------
    def get_tokenwise_summary_batches(
            self,
            scene_text: str,
            system_prompts: list[dict],
            SCENE_CONTEXT_THRESHOLD: float,
            prompt_manager,
            model_token_limit: int
        ) -> list[dict]:
        """
        Build evenly distributed batches for summarizing a full scene.

        Each batch includes:
            - Description (first batch)
            - Prior summary (subsequent batches)
            - Turns (shared roughly evenly across batches)

        Returns:
            - batch_text: full text for a single LLM user message
            - prior_summary_text: accumulated summary before this batch
            - turn_indices: list of turn numbers included
        """

        # --- Parse scene ---
        sections = self.parse_generic(scene_text)
        turns = self.extract_groups_from_sections(sections, header_regex=r"^Turn\s+(\d+)")

        # --- Calculate tokens per turn + total tokens ---
        turn_token_counts = []
        total_tokens = 0
        system_tokens = self.agent.count_tokens(system_prompts)

        # Include description tokens if present
        description_text = ""
        for sec in sections:
            if sec["header"] and sec["header"].lower() == "description":
                description_text = f"# Description\n{sec['text']}\n"
                break

        desc_tokens = self.agent.count_tokens_string(description_text) if description_text else 0

        for turn in turns:
            turn_text = ""
            for sec in turn["sections"]:
                if sec.get("header", "").lower() == "full turn":
                    turn_text = sec["text"]
                    break
            if not turn_text:
                turn_text = "\n".join(sec["text"] for sec in turn["sections"])

            turn_block = f"# Turn {turn['index']}\n{turn_text}\n"
            tokens = self.agent.count_tokens_string(turn_block)
            turn_token_counts.append(tokens)
            total_tokens += tokens

        total_tokens += system_tokens + desc_tokens

        # --- Determine number of batches needed ---
        max_batch_tokens = int(model_token_limit * SCENE_CONTEXT_THRESHOLD)
        num_batches = max(1, math.ceil(total_tokens / max_batch_tokens))

        # --- Distribute turns across batches evenly ---
        batches = []
        prior_summary_text = ""
        turn_idx = 0
        turns_per_batch = math.ceil(len(turns) / num_batches)

        for batch_num in range(num_batches):
            batch_lines = []

            # Include description only in first batch
            if batch_num == 0 and description_text:
                batch_lines.append(description_text)

            # Include prior summary for batches >= 2
            if prior_summary_text:
                summary_block = f"# Scene Summary\n{prior_summary_text}\n"
                batch_lines.append(summary_block)

            batch_turn_indices = []
            for _ in range(turns_per_batch):
                if turn_idx >= len(turns):
                    break
                turn = turns[turn_idx]
                turn_text = ""
                for sec in turn["sections"]:
                    if sec.get("header", "").lower() == "full turn":
                        turn_text = sec["text"]
                        break
                if not turn_text:
                    turn_text = "\n".join(sec["text"] for sec in turn["sections"])
                turn_block = f"# Turn {turn['index']}\n{turn_text}\n"
                batch_lines.append(turn_block)
                batch_turn_indices.append(turn["index"])
                turn_idx += 1

            batch_text = "\n".join(batch_lines).strip()
            batches.append({
                "batch_text": batch_text,
                "prior_summary_text": prior_summary_text,
                "turn_indices": batch_turn_indices,
            })

        return batches