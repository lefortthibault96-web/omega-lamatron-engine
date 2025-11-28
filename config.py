from pathlib import Path
from rich import print as rprint

DEFAULT_MODEL = "fluffy/l3-8b-stheno-v3.2"
MODEL = "dolphin3:8b"                   # optional, if you need both
DEFAULT_TEMPERATURE = 1.1               # how creative the character-playing LLM is allowed to be
SUMMARY_TEMPERATURE = 0.4               # how creative are summaries allowed to be (keep this lower than default for better results)
CONTEXT_THRESHOLD = 0.4                 # % of context for warnings and auto-summary (0-1)
SCENE_CONTEXT_THRESHOLD = 0.8           # % of context for scene summaries (0-1)
AUTO_SUMMARIZE = True                   # Automatically summarize when token usage is above context treshold
DYNAMIC_SUMMARY_ALLOW_MIDTURN = True    # True allows summaries mid-turn (as soon as threshold is exceeded), False will only summarize on next turn (/t)
FULL_TURNS_TO_KEEP = 1                  # How many last turns to leave unsummarized
MIN_SUMMARY_TURNS_TO_KEEP = 1           
MAX_SUMMARY_TURNS_TO_KEEP = 5
HELP_LINES = [
    "/h                   - Show help",
    "/r <dice>            - Roll dice",
    "/c                   - Combat submode",
    "/e                   - Exploration submode",
    "/r                   - Roleplay submode",
    "/g                   - Group submode",
    "/s                   - Summarize all unsummarized turns using full text and SCENE_CONTEXT_THRESHOLD",
    "try again <comment>  - Regenerate last LLM message (retry), optional comment is added to last user input as clarification for retries (cumulative)",
    "/ls                  - List characters",
    "/n                   - Next character",
    "/1 /2 /3             - Switch active character",
    "/t                   - Next turn",
    "*                    - Toggle auto-mode (when True, upon empty user input, switches to next character then sends)",
    ".                    - Append GM text in scene file without summoning LLM"
    "/end                 - End scene and launch a batched summary using full turn text (respecting scene context treshold)",
    "/scrunch             - DESTROYS Scene Summary, summarizing it to a single paragraph to clear accumulated context space",
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