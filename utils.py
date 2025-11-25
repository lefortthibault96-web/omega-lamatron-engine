from pathlib import Path
import subprocess
import re
from config import DEFAULT_MODEL, CONTEXT_THRESHOLD
from rich.console import Console

console = Console()

# ---------------------------------------------------------
# Path Helpers
# ---------------------------------------------------------
def safe_resolve(vault_root: Path, relative_path: str) -> Path:
    """
    Resolve a path inside the vault_root safely.
    Prevents directory traversal attacks.
    """
    candidate = (vault_root / relative_path).resolve()
    vault_root_res = vault_root.resolve()

    if not str(candidate).startswith(str(vault_root_res)):
        raise ValueError(f"Illegal path escape attempt: {relative_path}")

    return candidate


def read_vault_file(vault_root: Path, rel_path: str) -> str:
    """
    Reads text from a vault file using safe_resolve.
    Returns empty string if file does not exist.
    """
    p = safe_resolve(vault_root, rel_path)

    if not p.exists():
        return ""

    # Ensure parent directories always exist — useful in new vault setups
    p.parent.mkdir(parents=True, exist_ok=True)

    return p.read_text(encoding="utf-8")
    

# ---------------------------------------------------------
# Model Token Limit Helpers
# ---------------------------------------------------------
def get_model_token_limit(model_name: str) -> int:
    """
    Queries Ollama CLI `show` to get the model's context length (token limit).
    Returns None if CLI is missing or model info cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["ollama", "show", f"{model_name}:latest"],
            capture_output=True,
            text=True,
            check=True
        )

        # Look for line: context length      8192
        match = re.search(r"context length\s+(\d+)", result.stdout)
        if match:
            return int(match.group(1))
        else:
            console.print(
                f"[yellow][warning] Could not parse context length for {model_name}[/yellow]"
            )
            return None

    except Exception as e:
        console.print(
            f"[yellow][warning] Could not get token limit for {model_name}: {e}[/yellow]"
        )
        return None


DEFAULT_MODEL_TOKEN_LIMIT = get_model_token_limit(DEFAULT_MODEL)
console.print(
    f"[bold cyan]Token limit for {DEFAULT_MODEL}: {DEFAULT_MODEL_TOKEN_LIMIT}[/bold cyan]"
)


# ---------------------------------------------------------
# Context Usage Warning
# ---------------------------------------------------------
def check_context_usage(tokens_used: int, max_tokens: int):
    """
    Prints a warning if tokens_used exceeds your configurable CONTEXT_THRESHOLD.
    """
    if max_tokens is None:
        return  # not enough info to check

    usage_ratio = tokens_used / max_tokens

    if usage_ratio >= CONTEXT_THRESHOLD:
        percent = round(usage_ratio * 100, 1)
        console.print(
            f"[bold red]WARNING: {percent}% of context used[/bold red]"
        )
