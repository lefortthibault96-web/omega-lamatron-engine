from pathlib import Path
from rich.console import Console

from config import vault_root, scenes_active_dir, FULL_TURNS_TO_KEEP
from utils import read_vault_file, SceneDocument, get_turn_full_text, get_turn_summary_text

console = Console()


class PromptManager:
    """
    Central handler for:
      - loading prompt templates (cached)
      - building system/submode/group messages
      - building summary and memory prompts
      - building collapsed scene text through SceneDocument
    """

    def __init__(self, prompts_dir: Path, scene_text: str = ""):
        self.prompts_dir = prompts_dir
        self.cache = {}
        self.vault_root = vault_root
        self.scenes_active_dir = scenes_active_dir

    # ------------------------------------------------------------------
    # Scene Loading
    # ------------------------------------------------------------------
    def get_active_scene_file(self) -> Path | None:
        md_files = sorted(self.scenes_active_dir.glob("*.md"))
        return md_files[0] if md_files else None

    def load_scene(self) -> str:
        active_scene = self.get_active_scene_file()
        if not active_scene:
            return ""
        rel_path = str(active_scene.relative_to(self.vault_root))
        return read_vault_file(self.vault_root, rel_path)

    # ------------------------------------------------------------------
    # Cached Prompt Loading
    # ------------------------------------------------------------------
    def load(self, path: Path) -> str:
        if path in self.cache:
            return self.cache[path]
        if path.exists():
            content = path.read_text(encoding="utf-8")
        else:
            content = ""
        self.cache[path] = content
        return content

    # Base prompts
    def system_prompt(self) -> str:
        return self.load(self.prompts_dir / "system.md")

    def character_instructions(self) -> str:
        return self.load(self.prompts_dir / "character_instructions.md")

    def summary_prompt(self) -> str:
        return self.load(self.prompts_dir / "summary_system.md")

    def submode(self, mode: str) -> str:
        return self.load(self.prompts_dir / "submode" / f"{mode}.md")

    # ------------------------------------------------------------------
    # SUMMARY MESSAGES
    # ------------------------------------------------------------------
    def build_summary_messages(self, scene_text: str, prior_summary_text: str = "") -> list[dict]:
        """
        Build system+user messages to summarize a batch of turns.
        """
        if prior_summary_text:
            system_content = (
                f"{self.summary_prompt()}\n\n"
                f"Here is the previously written summary:\n{prior_summary_text}\n\n"
                "Do not rewrite this; continue from it as if the entire summary were written by one author.\n"
                "Summarize only the new batch text chronologically into ONE concise paragraph."
            )
        else:
            system_content = self.summary_prompt()

        user_content = (
            "Batch of scene text to summarize:\n"
            f"{scene_text}\n\n"
            "Your task:\n"
            "- Summarize ONLY this batch into ONE short paragraph (4–5 sentences max).\n"
            "- Start with the earliest event and proceed chronologically.\n"
            "- Only use information in this batch; do NOT invent new events.\n"
            "- No headings or bullet points.\n"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def build_turn_summary_messages(self, turn_text: str, turn_num: int) -> list[dict]:
        """
        Build a short, single-sentence-per-character summary message for a specific turn.
        """
        user_instruction = (
            f"Please write a short summary for Turn {turn_num}:\n"
            f"{turn_text}\n\n"
            "Write one sentence per character, summarizing factual events.\n"
            "No new information. Do not continue the story.\n"
        )

        return [
            {"role": "system", "content": self.summary_prompt()},
            {"role": "user", "content": user_instruction},
        ]

    # ------------------------------------------------------------------
    # MEMORY MESSAGES
    # ------------------------------------------------------------------
    def build_memory_messages(self, scene_text: str, char_name: str, char_sheet: str) -> list[dict]:
        """
        Build a one-sentence “memory” summary for a specific character.
        """
        system_block = f"{self.summary_prompt()}\n\n{char_sheet}"

        user_instruction = (
            f"Based on this scene:\n{scene_text}\n\n"
            f"Write ONE sentence describing the personal memory '{char_name}' will retain. "
            f"Include key events, people, and places relevant to this character."
        )

        return [
            {"role": "system", "content": system_block},
            {"role": "user", "content": user_instruction},
        ]

    # ------------------------------------------------------------------
    # COLLAPSED SCENE BUILDING (now uses SceneDocument instead of manual parsing)
    # ------------------------------------------------------------------
    def build_scene_text(self, turns_to_keep: int | None = None) -> str:
        """
        Build collapsed scene text using SceneDocument for clean extraction:
        - Use summaries for all but the last N turns.
        - Keep last N turns in full.
        """
        raw = self.load_scene()
        if not raw:
            return ""

        doc = SceneDocument(raw, lambda x: 0)

        if turns_to_keep is None:
            turns_to_keep = FULL_TURNS_TO_KEEP

        description = doc.description()
        turns = doc.turns  # list of {index: n, sections: [...]}

        full_turn_indices = [t["index"] for t in turns]
        if not full_turn_indices:
            return description

        cutoff = max(full_turn_indices) - turns_to_keep

        lines = []
        if description:
            lines.append("# Description")
            lines.append(description)
            lines.append("")

        for t in turns:
            idx = t["index"]
            lines.append(f"# Turn {idx}")

            # Old enough turn → try using summary
            if idx <= cutoff:
                summary = get_turn_summary_text(t)
                if summary:
                    lines.append(summary)
                    lines.append("")
                    continue

            # Otherwise → full text
            full = get_turn_full_text(t)
            if full:
                lines.append(full)
            lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # MAIN DIALOGUE MESSAGE BUILDERS
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

        # ---- Build unified system message ----
        system_content = (
            f"{system_prompt}\n\n"
            "### Character Instructions\n"
            f"{character_instructions}\n\n"
            "### Submode Instructions\n"
            f"{submode_instructions}\n\n"
            "### Active Character Sheet\n"
            f"{character_sheet}"
        )

        # ---- Build user message ----
        user_content = (
            f"Here is the scene so far:\n{scene_text}\n\n"
            f"You are {speaker_name}. Respond only as this character.\n"
            f"The GM says:\n{user_input}\n"
            f"Keep your answer short unless the GM requests more detail."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def build_group_messages(
        self,
        agent,
        system_prompt: str,
        scene_text: str,
        user_input: str,
    ) -> list[dict]:
        """
        Build full group prompt:
        - Loads a 'group.md' template.
        - Injects character sheets.
        """

        template_path = self.prompts_dir / "submode" / "group.md"
        template = self.load(template_path)

        # All character sheets
        sheet_blocks = []
        for name, path in zip(agent.character_names, agent.character_paths):
            sheet_text = read_vault_file(
                agent.vault_root, str(path.relative_to(agent.vault_root))
            )
            sheet_blocks.append(f"### CHARACTER: {name}\n{sheet_text}")

        combined_sheets = "\n\n".join(sheet_blocks)

        # Replace placeholders
        resolved = (
            template.replace("{{GROUP_NAMES}}", ", ".join(agent.character_names))
            .replace("{{CHARACTER_SHEETS}}", combined_sheets)
        )

        # System block
        system_content = f"{system_prompt}\n\n{resolved}"

        # User block
        user_content = (
            f"Here is the scene so far:\n{scene_text}\n\n"
            f"The GM asks the group:\n{user_input}"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
