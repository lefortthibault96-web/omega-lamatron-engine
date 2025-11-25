from pathlib import Path
import subprocess
import re
from rich.console import Console
console = Console()

DEFAULT_MODEL = "fluffy/l3-8b-stheno-v3.2"
MODEL = "dolphin3:8b"           # optional, if you need both
CONTEXT_THRESHOLD = 0.4        # % of context for warnings and auto-summary (0-1)
SCENE_CONTEXT_THRESHOLD = 0.5 # % of context for scene summaries (0-1)
SUMMARY_LIMIT_PER_BATCH = 100   # Max number of token the repo
AUTO_SUMMARIZE = True           # Automatically summarize when token usage is above context treshold
TURNS_TO_KEEP = 3               # How many last turns to leave unsummarized

BASE_DIR = Path(__file__).resolve().parent
# vault = Path(r"E:\\Users\\Tibo\\Obsidian\\PNJisme\\PNJisme\\Risus").resolve()
vault_root = Path(__file__).resolve().parent
characters_dir = vault_root / "Characters" / "Active"
scenes_active_dir = vault_root / "Scenes" / "Active"
prompts_dir = vault_root / "Prompts"

HELP_LINES = [
    "/h                   - Show help",
    "/r <dice>            - Roll dice",
    "/c                   - Combat submode",
    "/e                   - Exploration submode",
    "/r                   - Roleplay submode",
    "/g                   - Group submode",
    "/s <n>               - Summarize scene. Optionally keep the last N turns unsummarized (default uses config value)",
    "try again <comment>  - Regenerate last LLM message (retry), optional comment is added to last user input as clarification for retries (cumulative)",
    "/ls                  - List characters",
    "/n                   - Next character",
    "/1 /2 /3             - Switch active character",
    "/t                   - Next turn",
    "*                    - Toggle auto-mode (when True, upon empty user input, switches to next character then sends)",
    ".                    - Append GM text in scene file without summoning LLM"
    "/end                 - End scene and launch a batched summary using full turn text (respecting scene context treshold)",
]

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