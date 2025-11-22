from pathlib import Path
import subprocess
import re
from rich.console import Console
console = Console()

DEFAULT_MODEL = "fluffy/l3-8b-stheno-v3.2"
MODEL = "dolphin3:8b"  # optional, if you need both
CONTEXT_THRESHOLD = 0.1 # % of context for warnings and auto-summary (0-1)
AUTO_SUMMARIZE = True          # Automatically summarize when token usage is above context treshold
TURNS_TO_KEEP = 3              # How many last turns to leave unsummarized

BASE_DIR = Path(__file__).resolve().parent
# vault = Path(r"E:\\Users\\Tibo\\Obsidian\\PNJisme\\PNJisme\\Risus").resolve()
vault = Path(__file__).resolve().parent
characters = vault / "Characters" / "Active"
scenes_active = vault / "Scenes" / "Active"
prompts_dir = vault / "Prompts"

# Helper functions
def safe_resolve(vault_root: Path, relative_path: str) -> Path:
    candidate = (vault_root / relative_path).resolve()
    if not str(candidate).startswith(str(vault_root.resolve())):
        raise ValueError("Illegal path escape attempt")
    return candidate

def read_vault_file(vault_root: Path, rel_path: str) -> str:
    p = safe_resolve(vault_root, rel_path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")

def get_model_token_limit(model_name: str) -> int:
    """
    Queries Ollama CLI `show` to get the model's context length (token limit).
    Returns None if it cannot be determined.
    """
    try:
        result = subprocess.run(
            ["ollama", "show", f"{model_name}:latest"],
            capture_output=True,
            text=True,
            check=True
        )
        # Look for line like "context length      8192"
        match = re.search(r"context length\s+(\d+)", result.stdout)
        if match:
            return int(match.group(1))
        else:
            print(f"[warning] Could not parse context length for {model_name}")
            return None
    except Exception as e:
        print(f"[warning] Could not get token limit for {model_name}: {e}")
        return None

DEFAULT_MODEL = "fluffy/l3-8b-stheno-v3.2"
DEFAULT_MODEL_TOKEN_LIMIT = get_model_token_limit(DEFAULT_MODEL)
print(f"Token limit for {DEFAULT_MODEL}: {DEFAULT_MODEL_TOKEN_LIMIT}")

def check_context_usage(tokens_used: int, max_tokens: int):
    """
    Compare tokens used with max_tokens.
    Prints a warning if usage exceeds CONTEXT_THRESHOLD.
    """
    if max_tokens is None:
        return  # cannot check without max

    usage_ratio = tokens_used / max_tokens
    if usage_ratio >= CONTEXT_THRESHOLD:
        percent = round(usage_ratio * 100, 1)
        console.print(f"[bold red]WARNING: {percent}% of context used[/bold red]")