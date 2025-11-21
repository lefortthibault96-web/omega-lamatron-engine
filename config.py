from pathlib import Path

DEFAULT_MODEL = "fluffy/l3-8b-stheno-v3.2"
MODEL = "dolphin3:8b"  # optional, if you need both

vault = Path(r"E:\\Users\\Tibo\\Obsidian\\PNJisme\\PNJisme\\Risus").resolve()
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