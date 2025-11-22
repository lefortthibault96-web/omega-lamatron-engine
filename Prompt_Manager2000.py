from pathlib import Path
from rich.console import Console

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
        self.scene_text = scene_text  # Scene text is passed in externally

    # ------------------------------------------------------------------
    # Internal cached loader
    # ------------------------------------------------------------------
    def _load(self, path: Path) -> str:
        if path in self.cache:
            return self.cache[path]

        if not path.exists():
            console.print(f"[PromptManager] Missing prompt: {path}", style="bold yellow")
            self.cache[path] = ""
            return ""

        text = path.read_text(encoding="utf-8").strip()
        self.cache[path] = text
        return text

    # ------------------------------------------------------------------
    # Base prompt getters
    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        return self._load(self.prompts_dir / "system.md")

    def character_instructions(self) -> str:
        return self._load(self.prompts_dir / "character_instructions.md")

    def summary_prompt(self) -> str:
        return self._load(self.prompts_dir / "summary_system.md")

    def submode(self, mode: str) -> str:
        return self._load(self.prompts_dir / "submode" / f"{mode}.md")

    def load(self, filename: str) -> str:
        return self._load(self.prompts_dir / filename)

    # ------------------------------------------------------------------
    # SUMMARY MESSAGES
    # ------------------------------------------------------------------
    def build_summary_messages(self, scene_text: str) -> list[dict]:
        user_instruction = (
            "Please write a short summary of the following scene:\n"
            f"{scene_text}\n\n"
            "Write one concise paragraph mentioning key characters, events, and places. "
            "Do NOT continue the story or add new details. "
            "Do NOT use bullet lists."
        )
        return [
            {"role": "system", "content": self.summary_prompt()},
            {"role": "user", "content": user_instruction},
        ]

    def build_turn_summary_messages(self, turn_text: str, turn_num: int) -> list[dict]:
        user_instruction = (
            f"Please write a short summary for Turn {turn_num}:\n"
            f"{turn_text}\n\n"
            "Write one sentence per character, mentioning key characters, actions, and events that occurred in this turn. "
            "Do NOT continue the story beyond what is in this turn. "
            "Do NOT add bullet lists or new information. "
            "Focus on the outcomes of actions and the situation at the end rather than attempts."
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

    # ------------------------------------------------------------------
    # SCENE TEXT BUILDER
    # ------------------------------------------------------------------
    def build_scene_text(self, scene_text: str, turns_to_keep: int = 3) -> str:
        """
        Builds a collapsed scene text for LLM input:
        - Uses summaries if present (under '## Summary')
        - Keeps the last `turns_to_keep` turns fully detailed
        - Returns text suitable for LLM context
        """
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
            elif in_full_turn:
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
    def build_messages(
        self,
        system_prompt: str,
        character_instructions: str,
        submode_instructions: str,
        character_sheet: str,
        speaker_name: str,
        shortened_scene: str,
        user_input: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": character_instructions},
            {"role": "system", "content": submode_instructions},
            {"role": "system", "content": f"ACTIVE CHARACTER SHEET:\n{character_sheet}"},
            {"role": "user", "content": f"You are {speaker_name}. Respond only as this character."},
            {"role": "user", "content": (
                f"Here is the scene so far:\n{shortened_scene}\n\n"
                f"The GM has said to you:\n{user_input}\nKeep your answer short unless the GM explicitly requests length."
            )}
        ]

    def build_group_messages(
        self,
        system_prompt: str,
        group_prompt: str,
        shortened_scene: str,
        user_input: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": group_prompt},
            {"role": "user", "content": (
                f"Here is the context you are in:\n{shortened_scene}\n\n"
                f"The GM has asked:\n{user_input}\n"
            )},
        ]
