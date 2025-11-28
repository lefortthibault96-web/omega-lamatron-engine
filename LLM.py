import os
from pathlib import Path
import ollama
from ollama import generate
from rich.console import Console

from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from utils import safe_resolve, read_vault_file, write_vault_file

import tiktoken


console = Console()


# ---------------------------------------------------------
# OllamaAgent
# ---------------------------------------------------------
class OllamaAgent:
    """
    High-level wrapper for:
      - file-safe R/W inside vault_root
      - LLM interactions through Ollama
      - token counting
      - rollback of last writes via file offsets
      - character loading
    """

    def __init__(
        self,
        vault_root: Path,
        characters_dir: Path,
        scenes_active_dir: Path,
        model=DEFAULT_MODEL,
    ):
        self.client = ollama.Client(host="http://localhost:11434")
        self.model = model

        self.vault_root = vault_root
        self.characters_dir = characters_dir
        self.scenes_active_dir = scenes_active_dir

        self.auto_mode = False

        # Character scanning
        self.character_paths = sorted(characters_dir.glob("*.md"))
        self.character_names = [p.stem for p in self.character_paths]
        self.active_character_index = 0

        # File rollback tracking
        self._last_append = None
        self._last_llm_append = None

        # Conversation history (optional)
        self._conversation_history = []

        console.print("[Character Loading]", style="bold cyan")
        if self.character_names:
            for idx, name in enumerate(self.character_names, 1):
                console.print(
                    f"Loaded character [{idx}]: {name}", style="bold green"
                )
        else:
            console.print(
                "[Warning] No characters found in Characters/Active",
                style="bold yellow",
            )

    # ---------------------------------------------------------
    # Scene Access
    # ---------------------------------------------------------
    def get_active_scene_path(self):
        md_files = sorted(self.scenes_active_dir.glob("*.md"))
        return md_files[0] if md_files else None

    def read_active_scene(self):
        """
        Safe read via read_vault_file.
        """
        path = self.get_active_scene_path()
        if not path:
            return ""
        rel = str(path.relative_to(self.vault_root))
        return read_vault_file(self.vault_root, rel)

    # ---------------------------------------------------------
    # Append Helpers (Scene Writing with Rollback Tracking)
    # ---------------------------------------------------------
    def _append_raw(self, abs_path: Path, text: str):
        """
        Append raw text to a file, tracking offset for rollback.
        """
        payload = ("\n\n" + text.strip() + "\n").encode("utf-8")

        with open(abs_path, "ab+") as f:
            f.seek(0, os.SEEK_END)
            offset = f.tell()
            f.write(payload)
            f.flush()

        self._last_append = {
            "file": abs_path,
            "offset": offset,
            "length": len(payload),
        }

    def append_to_active_scene(self, text: str):
        """
        Append GM or LLM text directly to current scene, tracking for rollback.
        """
        path = self.get_active_scene_path()
        if not path:
            return False

        abs_path = safe_resolve(
            self.vault_root,
            str(path.relative_to(self.vault_root)),
        )

        self._append_raw(abs_path, text)
        return True

    def append_llm_output(self, text: str):
        """
        Append LLM output and track separately for "retry" rollback.
        """
        self.append_to_active_scene(text)

        if self._last_append:
            self._last_llm_append = self._last_append.copy()

        self.scene_raw = self.read_active_scene()
        return True

    # ---------------------------------------------------------
    # Rollback
    # ---------------------------------------------------------
    def rollback_last_llm_output(self):
        """
        Rollback ONLY last assistant message.
        """
        if not self._last_llm_append:
            return False

        entry = self._last_llm_append
        file = entry["file"]
        offset = entry["offset"]

        try:
            with open(file, "r+b") as f:
                f.truncate(offset)

            self.scene_raw = self.read_active_scene()
            self._last_llm_append = None
            self._last_append = None
            return True

        except Exception:
            return False

    def rollback_last_append(self):
        """
        Rollback last append, regardless of speaker.
        """
        if not self._last_append:
            return False

        entry = self._last_append
        file = entry["file"]
        offset = entry["offset"]

        try:
            with open(file, "r+b") as f:
                f.truncate(offset)

            self.scene_raw = self.read_active_scene()
            self._last_append = None
            return True

        except Exception:
            return False

    # ---------------------------------------------------------
    # Character Sheets
    # ---------------------------------------------------------
    def get_character_sheet_by_name(self, name: str) -> str:
        for path in self.character_paths:
            if path.stem == name:
                rel = str(path.relative_to(self.vault_root))
                return read_vault_file(self.vault_root, rel)
        return ""

    # ---------------------------------------------------------
    # LLM Chat (Ollama generate)
    # ---------------------------------------------------------
    def chat(self, messages: list[dict]) -> str:
        """
        High-level LLM call using Ollama's generate().
        """
        temperature = DEFAULT_TEMPERATURE

        system_prompt = "\n".join(
            msg["content"] for msg in messages if msg["role"] == "system"
        ).strip()

        user_prompt = "\n".join(
            msg["content"] for msg in messages if msg["role"] == "user"
        ).strip()

        response = generate(
            model=self.model,
            prompt=user_prompt,
            system=system_prompt,
            options={"temperature": temperature},
        )
        
        return response["response"].strip()

    # ---------------------------------------------------------
    # Token Counting
    # ---------------------------------------------------------
    def _get_encoding(self, model_to_use: str = None):
        if model_to_use is None:
            model_to_use = self.model

        try:
            return tiktoken.encoding_for_model(model_to_use)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(
        self,
        messages: list[dict] = None,
        model_to_use: str = None,
        return_breakdown: bool = False,
        include_history: bool = True,
    ):
        """
        Count tokens for messages (optionally including conversation history).
        Returns total + breakdown if requested.
        """

        encoding = self._get_encoding(model_to_use)
        all_messages = []

        if include_history and hasattr(self, "_conversation_history"):
            all_messages.extend(self._conversation_history)

        if messages:
            all_messages.extend(messages)

        total_tokens = 0
        breakdown = {}

        for msg in all_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            msg_tokens = len(encoding.encode(role)) + len(
                encoding.encode(content)
            ) + 4  # overhead

            total_tokens += msg_tokens
            breakdown[role] = breakdown.get(role, 0) + msg_tokens

        total_tokens += 2  # priming overhead

        if return_breakdown:
            return total_tokens, breakdown
        return total_tokens

    def count_tokens_string(self, text: str, model_to_use: str = None) -> int:
        encoding = self._get_encoding(model_to_use)
        return len(encoding.encode(text))
