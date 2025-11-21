# snitch.py
import re
from rich.console import Console

console = Console()

NUMBER_RE = re.compile(r"(\d+)")

class SnitchEditor:
    def __init__(self, sheet_text: str):
        self.sheet_text = sheet_text
        self.sheet_lines = []
        self.parse_sheet(sheet_text)

    def parse_sheet(self, text: str):
        """Parse Markdown sheet into structured lines with path info."""
        self.sheet_lines = []
        current_path = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Detect headers and update path
            if stripped.startswith("#"):
                level = stripped.count("#")
                header_text = stripped[level:].strip()
                current_path = current_path[: level - 1] + [header_text]

            # Store each line with its path
            self.sheet_lines.append({
                "line": stripped,
                "path": list(current_path)
            })

    def find_matches(self, keyword: str):
        """Find all lines containing the keyword, ignoring Markdown formatting."""
        matches = []
        keyword_lower = keyword.lower().strip()
        for idx, line_info in enumerate(self.sheet_lines):
            # Remove Markdown formatting for matching
            plain_line = re.sub(r"[_*`]", "", line_info["line"]).lower()
            if keyword_lower in plain_line:
                matches.append({
                    "line_idx": idx,
                    "line": line_info["line"],
                    "path": line_info["path"],
                    "context": " -> ".join(line_info["path"] + [line_info["line"]])
                })
        return matches


    def print_matches(self, matches):
        for i, m in enumerate(matches, 1):
            console.print(f"[Snitch] Match #{i}", style="bold yellow")
            console.print(f"  Path: {' -> '.join(m['path'])}")
            console.print(f"  Text: {m['line']}")

    def adjust_match(self, match, adjustment: int, min_value=0, max_value=99):
        """
        Adjust the first number found in the line or upward in path.
        Returns the modified line index, or None if no number found.
        """
        idx = match["line_idx"]
        # Search upward for a line with a number
        while idx >= 0:
            line_text = self.sheet_lines[idx]["line"]
            num_match = NUMBER_RE.search(line_text)
            if num_match:
                old_val = int(num_match.group(1))
                new_val = max(min_value, min(max_value, old_val + adjustment))
                # Replace first number in the line
                new_line = line_text[:num_match.start(1)] + str(new_val) + line_text[num_match.end(1):]
                self.sheet_lines[idx]["line"] = new_line
                console.print(f"[Snitch] Number adjusted: {line_text} -> {new_line}", style="bold green")
                return idx
            idx -= 1
        console.print("[Snitch] No number found to adjust", style="bold red")
        return None

    def to_text(self):
        """Reconstruct the Markdown sheet from sheet_lines."""
        return "\n".join(line["line"] for line in self.sheet_lines)


def write_vault_file(vault_root, rel_path, text):
    """Helper to write updated sheet back to file."""
    from pathlib import Path
    p = (vault_root / rel_path).resolve()
    if not str(p).startswith(str(vault_root.resolve())):
        raise ValueError("Illegal path escape attempt")
    p.write_text(text, encoding="utf-8")

def run_snitch_auto_detection(assistant_text, editor, console):
    """
    Run auto-detection on the assistant's output.
    Returns the list of hits (or empty list).
    """

    assistant_l = assistant_text.lower()
    hits = []

    for entry in editor.sheet_lines:
        raw_line = entry["line"].strip()
        if raw_line.startswith("#"):
            continue

        clean = re.sub(r"[*_`>#-]", "", raw_line).strip()
        if not clean:
            continue

        has_number = bool(re.search(r"\d+", clean))
        tokens = clean.split()
        candidates = []

        # Proper nouns
        for t in tokens:
            if t[:1].isupper() and len(t) > 2:
                candidates.append(t)

        # Multiword proper phrases
        capital_words = [t for t in tokens if t[:1].isupper()]
        if len(capital_words) >= 2:
            candidates.append(" ".join(capital_words))

        # Long nouns
        for t in tokens:
            if len(t) >= 8 and not t.lower().endswith(("ing", "ed")):
                candidates.append(t)

        if not candidates and not has_number:
            continue

        found = False

        if has_number and clean.lower() in assistant_l:
            found = True

        if not found:
            for c in candidates:
                if c.lower() in assistant_l:
                    found = True
                    break

        if found:
            m = editor.find_matches(clean)
            if m:
                hits.extend(m)

    # ----- Print first hit if any -----
    if hits:
        console.print("[Snitch_bitch]", style="bold yellow")
        editor.print_matches([hits[0]])
        editor._last_hits = hits

    return hits
