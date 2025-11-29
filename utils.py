# utils.py — Consolidated utility module for Lamatron
# Contains filesystem, markdown parsing, batching, turn utilities,
# scene inspection helpers, token helpers, and shared regex patterns.

from pathlib import Path
import subprocess
import re
from rich.console import Console
from config import DEFAULT_MODEL, CONTEXT_THRESHOLD

console = Console()

# ---------------------------------------------------------
# GLOBAL REGEX PATTERNS
# ---------------------------------------------------------
TURN_HEADER_RE = re.compile(r"^#{1,3}\s*Turn[: ]+(\d+)", re.I)
SCENE_SUMMARY_RE = re.compile(
    r"^#{1,3}\s*Scene Summary\s*\(Collapsed:\s*([0-9,\s\-]+)\)",
    re.M
)
TURN_IN_PROGRESS_RE = re.compile(
    r"(?ms)^#\s*Turn\s+\d+\s*\(In Progress\)"
)


# ---------------------------------------------------------
# PATH HELPERS
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

    p.parent.mkdir(parents=True, exist_ok=True)
    return p.read_text(encoding="utf-8")


def write_vault_file(vault_root: Path, rel_path: str, text: str):
    """
    Safe write using the vault root.
    Creates parent directories automatically.
    """
    p = safe_resolve(vault_root, rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------
# MODEL TOKEN LIMIT HELPERS
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


def check_context_usage(tokens_used: int, max_tokens: int):
    """
    Prints a warning if tokens_used exceeds your configurable CONTEXT_THRESHOLD.
    """
    if max_tokens is None:
        return

    usage_ratio = tokens_used / max_tokens

    if usage_ratio >= CONTEXT_THRESHOLD:
        percent = round(usage_ratio * 100, 1)
        console.print(
            f"[bold red]WARNING: {percent}% of context used[/bold red]"
        )


# ---------------------------------------------------------
# GENERAL TOKEN HELPERS
# ---------------------------------------------------------
def count_tokens_for_text_blocks(token_fn, *blocks):
    """
    Sum token counts for multiple text blocks using agent.count_tokens_string.
    """
    return sum(token_fn(b) for b in blocks if b)


def join_blocks(*blocks):
    """
    Join non-empty text blocks with double newlines.
    """
    return "\n\n".join(b for b in blocks if b).strip()


# ---------------------------------------------------------
# MARKDOWN PARSING
# ---------------------------------------------------------
def parse_sections(text, token_counter):
    """
    Parses text into sections of:
      - level (#=1, ##=2, etc.)
      - header
      - lines
      - text
      - tokens (via token_counter)
      - id (index number)
    """
    lines = text.splitlines()
    sections = []
    current = None
    header_re = re.compile(r"^(#+)\s+(.*)$")

    for line in lines:
        m = header_re.match(line.strip())
        if m:
            if current:
                current["text"] = "\n".join(current["lines"])
                current["tokens"] = token_counter(current["text"])
                sections.append(current)

            current = {
                "level": len(m.group(1)),
                "header": m.group(2).strip(),
                "lines": []
            }
        else:
            if current:
                # FIX: keep raw body lines of turns!
                current["lines"].append(line.rstrip())
            else:
                # ignore text before first header
                continue

    if current:
        current["text"] = "\n".join(current["lines"])
        current["tokens"] = token_counter(current["text"])
        sections.append(current)

    for i, sec in enumerate(sections):
        sec["id"] = i

    return sections


def extract_groups(sections, header_regex=r"^Turn\s+(\d+)", ignore_case=True):
    """
    Group sections by a header pattern (default: Turn X).
    Returns a list of:
        { index: turn_number, sections: [section objects] }
    """
    flags = re.IGNORECASE if ignore_case else 0
    group_re = re.compile(header_regex, flags)

    groups = []
    current = None

    for sec in sections:
        header = sec.get("header")
        m = group_re.match(header) if header else None

        if m:
            if current:
                groups.append(current)
            current = {"index": int(m.group(1)), "sections": []}

        if current:
            current["sections"].append(sec)

    if current:
        groups.append(current)

    return groups


# ---------------------------------------------------------
# SCENE CONTENT HELPERS
# ---------------------------------------------------------
def extract_description(sections):
    """
    Returns the raw description text if a # Description block exists.
    """
    for sec in sections:
        if (sec["header"] or "").lower() == "description":
            return sec["text"]
    return ""


def get_turn_full_text(turn_group):
    """
    Returns the 'Full Turn' text if present, else concatenation of all section texts.
    """
    for sec in turn_group["sections"]:
        if (sec.get("header") or "").lower() == "full turn":
            return sec["text"]
    return "\n".join(sec["text"] for sec in turn_group["sections"])


def get_turn_summary_text(turn_group):
    """
    Retrieve a turn's Summary text if present.
    """
    for sec in turn_group["sections"]:
        if (sec.get("header") or "").lower() == "summary":
            return sec["text"]
    return None


def has_turn_in_progress(scene_text: str) -> bool:
    """
    Detect if any turn is currently marked (In Progress).
    """
    return bool(TURN_IN_PROGRESS_RE.search(scene_text))


def parse_collapsed_turns(scene_text: str) -> set:
    """
    Parse collapsed turns from a Scene Summary (Collapsed: ...).
    Returns a set of integers.
    """
    m = SCENE_SUMMARY_RE.search(scene_text)
    collapsed = set()
    if m and m.group(1):
        for part in m.group(1).split(","):
            part = part.strip()
            if "-" in part:
                a, b = map(int, part.split("-"))
                collapsed.update(range(a, b + 1))
            else:
                collapsed.add(int(part))
    return collapsed


# ---------------------------------------------------------
# TURN CLEANING UTILITIES
# ---------------------------------------------------------
def remove_empty_turns(lines):
    """
    Remove empty Turn blocks. Returns the modified line list.
    """
    turn_positions = [
        (i, int(m.group(1))) for i, line in enumerate(lines)
        if (m := TURN_HEADER_RE.match(line))
    ]

    to_delete = []

    for idx, (line_idx, turn_num) in enumerate(turn_positions):
        next_idx = turn_positions[idx + 1][0] if idx + 1 < len(turn_positions) else len(lines)
        block = lines[line_idx:next_idx]

        content_lines = [
            l for l in block
            if not l.strip().lower().startswith("# turn")
            and not l.strip().lower().startswith("## summary")
            and not l.strip().lower().startswith("## full")
        ]

        if all(not l.strip() for l in content_lines):
            console.print(f"[yellow]Removed empty Turn {turn_num}[/yellow]")
            to_delete.append((line_idx, next_idx))

    for start, end in reversed(to_delete):
        del lines[start:end]

    return lines


def clean_scene_turns(text: str) -> str:
    """
    Clean turn blocks by removing empty ones and normalizing whitespace.
    """
    lines = text.splitlines()
    cleaned = remove_empty_turns(lines)
    return "\n".join(cleaned).strip() + "\n"


# ---------------------------------------------------------
# SCENE DOCUMENT WRAPPER (OBJECT-ORIENTED UTILITY)
# ---------------------------------------------------------
class SceneDocument:
    """
    A high-level parser for scene text.
    Unifies section parsing, turn grouping, summaries, full turns,
    collapsed turn metadata, and structural inspection.
    """

    def __init__(self, text, token_fn):
        self.text = text
        self.sections = parse_sections(text, token_fn)
        self.turns = extract_groups(self.sections)

    def description(self):
        return extract_description(self.sections)

    def full_turn(self, idx):
        for group in self.turns:
            if group["index"] == idx:
                return get_turn_full_text(group)
        return ""

    def summary(self, idx):
        for group in self.turns:
            if group["index"] == idx:
                return get_turn_summary_text(group)
        return None

    def collapsed_turns(self):
        return parse_collapsed_turns(self.text)

    def has_turn_in_progress(self):
        return has_turn_in_progress(self.text)

    def finished_turn_indices(self):
        """
        Returns list of turn numbers that have a full turn text.
        """
        finals = []
        for group in self.turns:
            if get_turn_full_text(group):
                finals.append(group["index"])
        return finals


# ---------------------------------------------------------
# BATCHING HELPERS (kept from original utils, unchanged)
# ---------------------------------------------------------
def compute_tokenwise_batches(blocks, system_tokens, threshold):
    allowed = threshold - system_tokens
    if allowed <= 0:
        raise ValueError("System tokens exceed threshold.")

    batches = []
    cur_blocks = []
    cur_toks = 0

    for b in blocks:
        bt = b["tokens"]
        if cur_blocks and cur_toks + bt > allowed:
            batches.append({
                "indices": [x["id"] for x in cur_blocks],
                "text": "\n\n".join(x["text"] for x in cur_blocks),
                "tokens": cur_toks
            })
            cur_blocks = []
            cur_toks = 0

        cur_blocks.append(b)
        cur_toks += bt

    if cur_blocks:
        batches.append({
            "indices": [x["id"] for x in cur_blocks],
            "text": "\n\n".join(x["text"] for x in cur_blocks),
            "tokens": cur_toks
        })

    return batches


def build_batch_text(batches, label_prefix="Block"):
    lines = []
    for b in batches:
        ids = b["indices"]
        start, end = ids[0], ids[-1]

        if start == end:
            lines.append(f"# {label_prefix} {start}")
        else:
            lines.append(f"# {label_prefix}s {start}–{end}")

        lines.append(b["text"])
        lines.append("")

    return "\n".join(lines).strip()
