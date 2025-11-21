#!/usr/bin/env python3
import re
from pathlib import Path
import msvcrt
from rich.console import Console
from snitch import SnitchEditor, write_vault_file, run_snitch_auto_detection
from dice import roll_dice
from LLM import OllamaAgent
from Prompt_Manager2000 import PromptManager
from config import DEFAULT_MODEL, safe_resolve, read_vault_file, vault, characters, scenes_active, prompts_dir

# ---------- Configuration ----------
active_char = None
user_input = ""
console = Console()

# ---------- File helpers ----------
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
# ---------- Command Processing ----------
class GMInterface:
    def __init__(self, agent: OllamaAgent, prompt_manager: PromptManager,
                current_submode: str = "roleplay"):
        
        self.agent = agent
        self.pm = prompt_manager

        self.current_submode = current_submode
        self.submode_text = self.pm.submode(current_submode)

        # Cached base prompts
        self.SYSTEM_PROMPT = self.pm.system_prompt()
        self.CHARACTER_INSTRUCTIONS = self.pm.character_instructions()

        self.retry_feedback = []

    # Existing methods like show_help, normalize_llm_output, list_characters, next_character, etc.

    # ------------------- NEW HELPER -------------------
    def _send_to_llm(self, user_input: str):
        """
        Send GM input to the LLM, build correct prompt through PromptManager,
        append output to scene, store messages for retry.
        """

        scene_before_input = self.agent.read_active_scene()
        speaker_name = self.agent.character_names[self.agent.active_character_index]

        # Append GM input to scene
        if user_input.strip():
            self.agent.append_to_active_scene(f"GM : {user_input}")

        # ----------- BUILD MESSAGE LIST THROUGH PromptManager -----------

        if self.current_submode == "group":
            group_prompt = self.pm.build_group_system_prompt(self.agent)
            messages = self.pm.build_group_messages(
                system_prompt=self.SYSTEM_PROMPT,
                group_prompt=group_prompt,
                scene_text=scene_before_input,
                user_input=user_input,
            )
            speaker_name = "Group"

        else:
            # Single-character mode
            active_char_path = self.agent.character_paths[self.agent.active_character_index]
            active_char_sheet = read_vault_file(
                self.agent.vault_root,
                str(active_char_path.relative_to(self.agent.vault_root))
            )

            messages = self.pm.build_messages(
                system_prompt=self.SYSTEM_PROMPT,
                character_instructions=self.CHARACTER_INSTRUCTIONS,
                submode_instructions=self.submode_text,
                character_sheet=active_char_sheet,
                scene_before_input=scene_before_input,
                user_input=user_input,
                speaker_name=speaker_name,
            )

        # -----------------------------------------------------------------

        # Store for retry
        self.agent._last_user_message_for_retry = messages
        self.agent._last_user_message_for_retry_user_text = user_input

        # LLM call
        response = self.agent.chat(messages)

        # Normalize & append
        char_response = self.normalize_llm_output(response, speaker_name)
        console.print(char_response)
        self.agent.append_llm_output(char_response)



    # ------------------- AUTO NEXT -------------------
    def auto_next_character(self):
        self.next_character()
        self._send_to_llm("")

    def show_help(self):
        console.print("""
[bold cyan]Available Commands:[/bold cyan]

/h       - Show help
/r <dice>- Roll dice
/c       - Combat submode
/e       - Exploration submode
/r       - Roleplay submode
/s       - Summarize scene
/p       - Regenerate last LLM message (retry)
/b       - Rollback last GM message
/ls      - List characters
/n       - Next character
/1 /2 /3 - Switch active character
""")

    def normalize_llm_output(self, response: str, speaker_name: str) -> str:
        """
        Strip any leading prefix if it looks like the speaker name or part of it,
        followed by a colon, then prepend the canonical speaker_name once.
        """
        response = response.lstrip()

        # Split the speaker name into words
        name_parts = speaker_name.split()
        # Build regex: match any starting substring of the full name followed by optional whitespace and colon
        # E.g., "Professor", "Professor Whizzlebum", "Whizzlebum"
        patterns = []
        for i in range(len(name_parts)):
            sub_name = " ".join(name_parts[i:])  # take suffixes
            patterns.append(re.escape(sub_name))
        pattern = rf"^(?:{'|'.join(patterns)})\s*:"

        # Remove the prefix if it exists
        response = re.sub(pattern, '', response, count=1).lstrip()

        # Prepend canonical speaker
        return f"{speaker_name} : {response}"



    # --- NEW: list characters
    def list_characters(self):
        console.print("\n[bold cyan]Characters:[/bold cyan]")
        for i, name in enumerate(self.agent.character_names):
            marker = "[active]" if i == self.agent.active_character_index else ""
            console.print(f"{i+1}. {name} {marker}")
        console.print("")

    # --- NEW: switch active character by index
    def switch_character(self, idx: int):
        if 1 <= idx <= len(self.agent.character_names):
            self.agent.active_character_index = idx - 1
            console.print(f"[green]Active character is now: {self.agent.character_names[idx-1]}[/green]")
        else:
            console.print("[red]Invalid character index[/red]")

    # --- NEW: next character with wraparound
    def next_character(self):
        if not self.agent.character_names:
            console.print("[red]No characters to switch[/red]")
            return
        self.agent.active_character_index = (self.agent.active_character_index + 1) % len(self.agent.character_names)
        console.print(f"[green]Active character is now: {self.agent.character_names[self.agent.active_character_index]}[/green]")

    def handle_roll(self, expr: str):
        try:
            result = roll_dice(expr)
            console.print(f"[bold green]Roll: {expr}[/bold green]")
            if result["rolls"]:
                console.print(f"Rolls: {result['rolls']}")
            if "breakdown" in result:
                console.print(f"Breakdown: {result['breakdown']}")
            console.print(f"Total: [bold yellow]{result['total']}[/bold yellow]")
        except:
            console.print("[red]Invalid dice expression[/red]")


    def summarize_scene(self):
        scene_text = self.agent.read_active_scene()
        if not scene_text.strip():
            console.print("[No scene text to summarize]", style="bold red")
            return

        # ----- LLM summary using summary_system.md -----

        console.print("[cyan]Summarizing scene…[/cyan]")
        summary_messages = self.pm.build_summary_messages(scene_text)
        scene_summary = self.agent.chat(summary_messages).strip()

        # ----- Write updated scene -----
        new_scene_text = f"# Summary :\n{scene_summary}\n\n# Details :\n{scene_text}"

        abs_scene_path = self.agent.get_active_scene_path()
        if abs_scene_path:
            abs_scene_path.write_text(new_scene_text, encoding="utf-8")
            console.print("[bold cyan]Scene summary added to top.[/bold cyan]")
        else:
            console.print("[red]No active scene file found.[/red]")
            return

        # ----- Per-character memories -----
        for char_path in self.agent.character_paths:
            char_name = char_path.stem
            char_sheet = read_vault_file(
                self.agent.vault_root,
                str(char_path.relative_to(self.agent.vault_root))
            )

            console.print(f"[cyan]Generating memory for {char_name}…[/cyan]")
            memory_messages = self.pm.build_memory_messages(scene_text, char_name, char_sheet)
            memory_sentence = self.agent.chat(memory_messages).strip()

            # Insert the memory into the character sheet
            lines = char_sheet.splitlines()

            memory_indices = [
                (i, line) for i, line in enumerate(lines)
                if re.match(r"#+\s*memories?", line, re.I)
            ]

            if memory_indices:
                memory_indices.sort(key=lambda x: len(re.match(r"(#+)", x[1]).group(1)))
                header_index = memory_indices[0][0]

                insert_index = header_index + 1
                while insert_index < len(lines) and not lines[insert_index].startswith("#"):
                    insert_index += 1

                lines.insert(insert_index, f"- {memory_sentence}")
            else:
                lines.append("\n# Memories")
                lines.append(f"- {memory_sentence}")

            write_vault_file(
                self.agent.vault_root,
                char_path.relative_to(self.agent.vault_root),
                "\n".join(lines)
            )

            console.print(f"[bold green]Memory updated for {char_name}[/bold green]")

        console.print("[bold cyan]Scene summary and character memories updated successfully.[/bold cyan]")

    def regenerate_last(self):
        # Grab the last messages exactly as stored
        messages = getattr(self.agent, "_last_user_message_for_retry", None)
        if not messages:
            console.print("[yellow]Nothing to retry[/yellow]")
            return

        # Rollback last LLM output first
        rolled = self.agent.rollback_last_llm_output()
        if not rolled:
            console.print("[red]No previous LLM output to rollback.[/red]")

        # Merge cumulative retry feedback into the last user message
        if self.retry_feedback:
            # Find the last user message (should be the GM input + scene)
            for msg in reversed(messages):
                if msg["role"] == "user":
                    # Merge ALL feedback cumulatively
                    msg["content"] += "\n\nGM FEEDBACK FOR RETRY (cumulative):\n" + "\n".join(self.retry_feedback)
                    break

        console.print("[cyan]Regenerating last LLM response…[/cyan]")
        response = self.agent.chat(messages)

        # Normalize and append
        speaker_name = self.agent.character_names[self.agent.active_character_index]
        char_response = self.normalize_llm_output(response, speaker_name)
        self.agent.append_llm_output(char_response)

        console.print("\n[bold green]Updated Response:[/bold green]")
        console.print(char_response + "\n")

    # ---------- MAIN INTERACTIVE LOOP ----------
    def run(self):
        console.print("[bold cyan]GM Assistant Ready.[/bold cyan]\n")

        while True:
            user_input = input(f"\n({self.current_submode}) {self.agent.character_names[self.agent.active_character_index]} GM> ").strip()
            speaker_name = self.agent.character_names[self.agent.active_character_index]

            # -----------------------------------------------------
            # 1) Handle TRY AGAIN and /p BEFORE any other commands
            # -----------------------------------------------------
            if user_input.lower().startswith("try again") or user_input.startswith("/p"):
                parts = user_input.split(" ", 1)
                feedback = parts[1].strip() if len(parts) > 1 else ""

                if feedback:
                    self.retry_feedback.append(feedback)

                self.regenerate_last()
                continue
            # -----------------------------------------------------
                # Narration-only append
            if user_input.startswith("."):
                narration_text = "GM : " + user_input[1:].lstrip()
                self.agent.append_to_active_scene(narration_text)
                console.print(f"[Narration appended]", style="bold cyan")
                continue  # skip LLM response

                   # ------------------ Toggle auto mode ------------------
            if user_input == "*":
                self.auto_mode = not getattr(self, "auto_mode", False)
                status = "activated" if self.auto_mode else "deactivated"
                console.print(f"[bold cyan]Auto mode {status}[/bold cyan]")
                continue

            if user_input.startswith("/"):
                # Help
                if user_input == "/h":
                    self.show_help()
                    continue

                # Roll dice
                elif user_input.startswith("/r "):
                    self.handle_roll(user_input[3:].strip())
                    continue

                # Summarize scene
                elif user_input == "/s":
                    self.summarize_scene()
                    continue


                # List characters
                elif user_input == "/ls":
                    self.list_characters()
                    continue

                # Next character
                elif user_input == "/n":
                    self.next_character()
                    continue

                # Submode shortcuts (/c, /r, /e)
                elif user_input.lower() in ("/c", "/r", "/g", "/e"):
                    if user_input.lower() == "/c":
                        self.current_submode = "combat"
                    elif user_input.lower() == "/r":
                        self.current_submode = "roleplay"
                    elif user_input.lower() == "/g":
                        self.current_submode = "group"
                        self.submode_text = ""  # group mode manages its own system messages
                        continue
                    else:
                        self.current_submode = "exploration"
                    self.submode_text = self.pm.submode(self.current_submode)
                    console.print(f"[bold cyan]Submode switched to {self.current_submode}[/bold cyan]")
                    continue

                # Switch active character by number (/1 /2 /3)
                elif user_input[1:].isdigit():
                    self.switch_character(int(user_input[1:]))
                    continue

                # Unknown command
                else:
                    console.print("[yellow]Unknown command.[/yellow]")
                    continue

            # Empty input → treat as "GM says nothing" and continue the scene
            if not user_input:
                if getattr(self, "auto_mode", False):
                    self.auto_next_character()   # switches character AND sends empty input
                else:
                    self._send_to_llm("")        # just sends empty input
                continue


                # NORMAL FLOW: GM provides input → LLM responds → append to scene

            # Use the helper to send GM input to LLM and append
            self._send_to_llm(user_input)

    # ---------- Main ----------
def main():
    # Create PromptManager
    pm = PromptManager(prompts_dir)

    agent = OllamaAgent(vault, characters, scenes_active)

    gm = GMInterface(
        agent=agent,
        prompt_manager=pm,
        current_submode="roleplay",
    )

    gm.run()

if __name__ == "__main__":
    main()
