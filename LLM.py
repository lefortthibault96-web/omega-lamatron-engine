import os
from pathlib import Path
import ollama
from rich.console import Console
from config import DEFAULT_MODEL, safe_resolve, read_vault_file
import tiktoken

console = Console()
# ---------- Agent ----------
class OllamaAgent:
    def __init__(self, vault_root: Path, characters_dir: Path, scenes_active_dir: Path, model=DEFAULT_MODEL):
        self.client = ollama.Client(host="http://localhost:11434")
        self.model = model
        self.vault_root = vault_root
        self.characters_dir = characters_dir
        self.scenes_active_dir = scenes_active_dir
        self.auto_mode = False

        # --- NEW: scan characters and store paths/names
        self.character_paths = sorted(characters_dir.glob("*.md"))
        self.character_names = [p.stem for p in self.character_paths]
        self.active_character_index = 0
        self._last_append = None

        # NEW: Track last LLM output and GM input for retry
        self._last_llm_append = None
        self._last_user_message_for_retry = None

        # Startup load printout
        console.print("[Character Loading]", style="bold cyan")
        if self.character_names:
            for idx, name in enumerate(self.character_names, 1):
                console.print(f"Loaded character [{idx}]: {name}", style="bold green")
        else:
            console.print("[Warning] No characters found in Characters/Active", style="bold yellow")

    def get_active_scene_path(self):
        md_files = sorted(self.scenes_active_dir.glob("*.md"))
        return md_files[0] if md_files else None

    def read_active_scene(self):
        path = self.get_active_scene_path()
        if not path:
            return ""
        return read_vault_file(self.vault_root, str(path.relative_to(self.vault_root)))

    def append_to_active_scene(self, text: str):
        path = self.get_active_scene_path()
        if not path:
            return False

        abs_path = safe_resolve(self.vault_root, str(path.relative_to(self.vault_root)))

        payload = ("\n\n" + text.strip() + "\n").encode("utf-8")

        with open(abs_path, "ab+") as f:
            f.seek(0, os.SEEK_END)
            offset = f.tell()
            f.write(payload)
            f.flush()

        # Store BOTH offset and length
        self._last_append = {
            "file": abs_path,
            "offset": offset,
            "length": len(payload)
        }
        return True

    # --- NEW: Append LLM output and track for rollback using file offsets
    def append_llm_output(self, text: str):
        self.append_to_active_scene(text)
        if self._last_append:
            self._last_llm_append = {
                "file": self._last_append["file"],
                "offset": self._last_append["offset"],
                "length": self._last_append["length"]
            }
        return True

    # --- NEW: Rollback last LLM output (uses stored file+offset)
    def rollback_last_llm_output(self):
        if not self._last_llm_append:
            return False

        file = self._last_llm_append["file"]
        offset = self._last_llm_append["offset"]

        try:
            with open(file, "r+b") as f:
                f.truncate(offset)
            self._last_llm_append = None
            return True
        except Exception:
            return False

    def rollback_last_append(self):
        if not self._last_append:
            return False
        file = self._last_append["file"]
        offset = self._last_append["offset"]
        try:
            with open(file, "r+b") as f:
                f.truncate(offset)
            self._last_append = None
            return True
        except:
            return False

    def get_character_sheet_by_name(self, name: str) -> str:
        for path in self.character_paths:
            if path.stem == name:
                return read_vault_file(self.vault_root, str(path.relative_to(self.vault_root)))
        return ""

    def chat(self, messages):
        resp = self.client.chat(model=self.model, messages=messages)
        msg = None
        if isinstance(resp, dict):
            msg = resp.get("message")
            if hasattr(msg, "content"):
                return str(msg.content).strip()
            if isinstance(resp, dict):
                return str(resp.get("content", ""))
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return str(resp.message.content).strip()
        return str(resp).strip()
    

    def count_tokens(
        self,
        messages: list[dict] = None,
        model_to_use: str = None,
        return_breakdown: bool = False,
        include_history: bool = True
    ):
        """
        Count tokens for a list of messages, optionally including assistant messages from history.
        Returns total tokens and breakdown per role.
        """
        import tiktoken

        if model_to_use is None:
            model_to_use = self.model

        try:
            encoding = tiktoken.encoding_for_model(model_to_use)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        # Gather messages
        all_messages = messages.copy() if messages else []

        if include_history:
            # Include previous assistant outputs if available
            history = getattr(self, "_conversation_history", [])
            all_messages = history + all_messages

        total_tokens = 0
        breakdown = {}

        for msg in all_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            msg_tokens = len(encoding.encode(role)) + len(encoding.encode(content)) + 4  # message overhead
            total_tokens += msg_tokens
            breakdown[role] = breakdown.get(role, 0) + msg_tokens

        total_tokens += 2  # priming overhead

        if return_breakdown:
            return total_tokens, breakdown
        return total_tokens