from pathlib import Path
from rich import print as rprint

DEFAULT_MODEL = "fluffy/l3-8b-stheno-v3.2"
MODEL = "dolphin3:8b"           # optional, if you need both
CONTEXT_THRESHOLD = 0.4        # % of context for warnings and auto-summary (0-1)
SCENE_CONTEXT_THRESHOLD = 0.5 # % of context for scene summaries (0-1)
AUTO_SUMMARIZE = True           # Automatically summarize when token usage is above context treshold
TURNS_TO_KEEP = 3               # How many last turns to leave unsummarized
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
# ---------------------------------------------------------
# Vault folders
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
vault_root = BASE_DIR

characters_root = vault_root / "Characters"
characters_dir = characters_root / "Active"

scenes_root = vault_root / "Scenes"
scenes_active_dir = scenes_root / "Active"

prompts_dir = vault_root / "Prompts"

# ---------------------------------------------------------
# Ensure required directories exist
# ---------------------------------------------------------
REQUIRED_FOLDERS = [
    characters_root,
    characters_dir,
    scenes_root,
    scenes_active_dir,
    prompts_dir,
]

for folder in REQUIRED_FOLDERS:
    if not folder.exists():                 # <-- folder did NOT exist
        folder.mkdir(parents=True, exist_ok=True)
        rprint(f"[green][config][/green] Created folder: [cyan]{folder}[/cyan]")

# ---------------------------------------------------------
# Ensure at least one active scene file exists
# ---------------------------------------------------------
def ensure_active_scene_exists():
    md_files = list(scenes_active_dir.glob("*.md"))
    if not md_files:
        default_scene = scenes_active_dir / "scene1.md"
        default_scene.write_text(
            "# Description\n\nStart your adventure here.\n", encoding="utf-8"
        )
        rprint(f"[green][config][/green] Created default active scene: [cyan]{default_scene}[/cyan]")
        return default_scene
    return md_files[0]

ACTIVE_SCENE_FILE = ensure_active_scene_exists()