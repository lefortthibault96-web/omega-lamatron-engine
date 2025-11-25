from pathlib import Path
from rich.console import Console
from config import vault_root, scenes_active_dir
from utils import read_vault_file

console = Console()

class PromptManager:
    """
    Centralized handler for:
      - prompt file loading (cached)
      - submodes
      - summary and memory message building
      - system/character/submode message assembly
      - group prompt building
    """

    def __init__(self, prompts_dir: Path, scene_text: str = ""):
        self.prompts_dir = prompts_dir
        self.cache = {}
        self.vault_root = vault_root
        self.scenes_active_dir = scenes_active_dir

    # ------------------------------------------------------------------
    # Internal cached loader
    # ------------------------------------------------------------------
    def get_active_scene_file(self) -> Path | None:
        """
        Return the first active scene file in the active scenes directory.
        """
        md_files = sorted(self.scenes_active_dir.glob("*.md"))
        return md_files[0] if md_files else None


    def load_scene(self) -> str:
        """
        Always load the latest version of the active scene file.
        """
        active_scene = self.get_active_scene_file()
        if not active_scene:
            return ""
        rel_path = str(active_scene.relative_to(self.vault_root))
        return read_vault_file(self.vault_root, rel_path)


# ------------------------------------------------------------------
# Base prompt getters
# ------------------------------------------------------------------
    def system_prompt(self) -> str:
        return self.load(self.prompts_dir / "system.md")


    def character_instructions(self) -> str:
        return self.load(self.prompts_dir / "character_instructions.md")


    def summary_prompt(self) -> str:
        return self.load(self.prompts_dir / "summary_system.md")


    def submode(self, mode: str) -> str:
        return self.load(self.prompts_dir / "submode" / f"{mode}.md")

    def load(self, path: Path) -> str:
        """
        Load a file from the given path with caching.
        """
        if path in self.cache:
            return self.cache[path]
        if not path.exists():
            content = ""
        else:
            content = path.read_text(encoding="utf-8")
        self.cache[path] = content
        return content
    

    # ------------------------------------------------------------------
    # SUMMARY MESSAGES
    # ------------------------------------------------------------------
    def build_summary_messages(self, scene_text: str, prior_summary_text: str = "") -> list[dict]:
        """
        Returns a list of messages for summarizing a batch.
        Forces a concise, one-paragraph summary, oldest → newest turn.
        Prior summary is included only as context, never to be re-summarized.
        """

        if prior_summary_text:
            system_content = (
                f"{self.summary_prompt()}\n\n"
                f"Here's the text you wrote previously:\n{prior_summary_text}\n"
                "Continue writing right after these events, without mentionning them.\n"
                "Start with the earliest event from the batch (description or first turn) and continue chronologically.\n"
                
            )
            
        else:
            system_content = self.summary_prompt()

        user_instruction = (
                f"Batch of scene text to summarize:\n{scene_text}\n\n"
                "Your task:\n"
                "- Summarize ONLY the events in this batch into ONE concise paragraph.\n"
                "- Start with the earliest event (description or first turn) and continue chronologically.\n"
                "- Limit to 4–5 sentences (roughly 100–120 words).\n"
                "- Only include factual events from this batch; do NOT invent new details.\n"
                "- Do NOT use headings, bullet points, or commentary.\n"
                "- Focus on the important events happening to the characters, being knocked inconscious is an important event.\n"
                "Template:\n"
                "[Insert ONE short paragraph summarizing ONLY the events in this batch, in chronological order.]"
            )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_instruction},
        ]

    def build_turn_summary_messages(self, turn_text: str, turn_num: int) -> list[dict]:
        user_instruction = (
            f"Please write a short summary for Turn {turn_num}:\n"
            f"{turn_text}\n\n"
            "Write one sentence per character, mentioning key characters, actions, and events that occurred in this turn.\n"
            "Do NOT continue the story beyond what is in this turn.\n"
            "Do NOT add bullet lists or new information.\n"
            "Focus on the outcomes of actions and the situation at the end rather than attempts.\n"
        )
        return [
            {"role": "system", "content": self.summary_prompt()},
            {"role": "user", "content": user_instruction},
        ]

    # ------------------------------------------------------------------
    # MEMORY MESSAGES
    # ------------------------------------------------------------------
    def build_memory_messages(self, scene_text: str, char_name: str, char_sheet: str) -> list[dict]:
        system_block = self.summary_prompt() + "\n\n" + char_sheet
        user_instruction = (
            f"Based on this scene:\n{scene_text}\n\n"
            f"Write a single-sentence memory describing what the character '{char_name}' "
            f"will personally remember. Mention key events, people, and places."
        )
        return [
            {"role": "system", "content": system_block},
            {"role": "user", "content": user_instruction},
        ]
    
    def build_scene_text(self, turns_to_keep: int | None = None) -> str:
        """
        Builds a collapsed scene text for LLM input:
        - Uses summaries if present (under '## Summary')
        - Keeps the last `turns_to_keep` turns fully detailed
        - Returns text suitable for LLM context
        """
        scene_text = self.load_scene()
        if turns_to_keep is None:
            from config import TURNS_TO_KEEP
            turns_to_keep = TURNS_TO_KEEP
        lines = scene_text.splitlines()
        description_lines = []
        turns = []  # list of dicts: {"summary": [], "lines": []}

        current_summary_lines = []
        current_turn_lines = []
        in_description = False
        in_turn = False
        in_summary = False
        in_full_turn = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("# Description"):
                in_description = True
                in_turn = in_summary = in_full_turn = False
                continue

            elif stripped.startswith("# Turn"):
                # save previous turn
                if current_summary_lines or current_turn_lines:
                    turns.append({
                        "summary": current_summary_lines if current_summary_lines else None,
                        "lines": current_turn_lines
                    })
                current_summary_lines = []
                current_turn_lines = []
                in_description = False
                in_turn = True
                in_summary = False
                in_full_turn = False
                continue

            elif stripped.startswith("## Summary"):
                in_summary = True
                in_full_turn = False
                continue

            elif stripped.startswith("## Full Turn"):
                in_full_turn = True
                in_summary = False
                continue

            if in_description:
                description_lines.append(stripped)

            elif in_summary:
                current_summary_lines.append(stripped)

            elif in_full_turn or in_turn:
                # KEY FIX: keep content even without subheaders
                current_turn_lines.append(stripped)

        # append the last turn
        if current_summary_lines or current_turn_lines:
            turns.append({
                "summary": current_summary_lines if current_summary_lines else None,
                "lines": current_turn_lines
            })

        # determine cutoff: last N turns to keep fully detailed
        num_turns = len(turns)
        cutoff_idx = max(0, num_turns - turns_to_keep)

        # rebuild scene using summaries for older turns
        final_lines = description_lines + [""] if description_lines else []

        for idx, turn in enumerate(turns):
            final_lines.append(f"# Turn {idx + 1}")
            if idx < cutoff_idx and turn["summary"]:
                final_lines.extend(turn["summary"])
            else:
                final_lines.extend(turn["lines"])

        return "\n".join(final_lines)

    # ------------------------------------------------------------------
    # MAIN DIALOGUE MESSAGES
    # ------------------------------------------------------------------
    def build_single_character_messages(
        self,
        system_prompt: str,
        character_instructions: str,
        submode_instructions: str,
        character_sheet: str,
        speaker_name: str,
        scene_text: str,
        user_input: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": character_instructions},
            {"role": "system", "content": submode_instructions},
            {"role": "system", "content": f"ACTIVE CHARACTER SHEET:\n{character_sheet}"},
            {"role": "user", "content": (
                f"Here is the scene so far:\n{scene_text}\n\n"
                f"You are {speaker_name}. Respond only as this character. The GM has said to you:\n{user_input}\nKeep your answer short unless the GM explicitly requests length."
            )}
        ]

    def build_group_messages(
        self,
        agent,
        system_prompt: str,
        scene_text: str,
        user_input: str
    ) -> list[dict]:
        """
        Build messages for group mode:
        - System message: system prompt + group prompt loaded from file + all character sheets
        - User message: collapsed scene + GM input
        """

        # Load group prompt template from file
        group_template_path = self.prompts_dir / "submode" / "group.md"
        template = self.load(group_template_path)  # assuming self.load reads file text

        # Build character sheets text
        sheet_blocks = []
        for name, path in zip(agent.character_names, agent.character_paths):
            sheet_text = read_vault_file(agent.vault_root, str(path.relative_to(agent.vault_root)))
            sheet_blocks.append(f"### CHARACTER: {name}\n{sheet_text}")
        combined_sheets = "\n\n".join(sheet_blocks)

        # Replace placeholders
        group_prompt_text = template.replace("{{GROUP_NAMES}}", ", ".join(agent.character_names))
        group_prompt_text = group_prompt_text.replace("{{CHARACTER_SHEETS}}", combined_sheets)

        # Full system message: system prompt + group prompt
        full_system_content = f"{system_prompt}\n\n{group_prompt_text}"

        # User message: collapsed scene + GM input
        user_content = f"Here is the scene you are in:\n{scene_text}\n\nThe GM has asked:\n{user_input}"

        return [
            {"role": "system", "content": full_system_content},
            {"role": "user", "content": user_content},
        ]

