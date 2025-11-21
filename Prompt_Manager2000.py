from pathlib import Path
from rich.console import Console
from config import read_vault_file   # adjust import to your structure

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

    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self.cache = {}

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
    # GROUP SYSTEM PROMPT
    # ------------------------------------------------------------------
    def build_group_system_prompt(self, agent) -> str:
        group_template_path = self.prompts_dir / "submode" / "group.md"
        template = self._load(group_template_path)

        # Build character sheet blocks
        blocks = []
        for name, path in zip(agent.character_names, agent.character_paths):
            sheet_text = read_vault_file(agent.vault_root, str(path.relative_to(agent.vault_root)))
            blocks.append(f"### CHARACTER: {name}\n{sheet_text}")

        combined_sheets = "\n\n".join(blocks)
        names = ", ".join(agent.character_names)

        # Replace placeholders
        return (
            template
            .replace("{{GROUP_NAMES}}", names)
            .replace("{{CHARACTER_SHEETS}}", combined_sheets)
        )

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
    # MAIN DIALOGUE MESSAGES
    # ------------------------------------------------------------------
    def build_messages(
        self,
        system_prompt: str,
        character_instructions: str,
        submode_instructions: str,
        character_sheet: str,
        speaker_name: str,
        scene_before_input: str,
        user_input: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": character_instructions},
            {"role": "system", "content": submode_instructions},
            {"role": "system", "content": f"ACTIVE CHARACTER SHEET:\n{character_sheet}"},
            {"role": "user", "content": f"You are {speaker_name}. Respond only as this character."},
            {"role": "user", "content": (
                f"Here is the scene so far:\n{scene_before_input}\n\n"
                f"The GM has said to you:\n{user_input}\nKeep your answer short unless the GM explicitly requests length."
            )}
        ]

    def build_group_messages(self, system_prompt: str, group_prompt: str, scene_text: str, user_input: str) -> list[dict]:
        """
        Build the full message list for group submode.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": group_prompt},
            {"role": "user", "content": (
                f"Here is the context you are in:\n{scene_text}\n\n"
                f"The GM has asked:\n{user_input}\n"
            )},
        ]