#!/usr/bin/env python3
import re
from pathlib import Path
import msvcrt
from rich.console import Console
from snitch import SnitchEditor, write_vault_file, run_snitch_auto_detection
from dice import roll_dice
from LLM import OllamaAgent
from Prompt_Manager2000 import PromptManager
from config import safe_resolve, read_vault_file, vault_root, characters_dir, scenes_active_dir, prompts_dir, HELP_LINES, SCENE_CONTEXT_THRESHOLD, DEFAULT_MODEL_TOKEN_LIMIT
from turns import ensure_current_turn, advance_turn, summarize_scene_turns
from batch import BatchManager

# ---------- Configuration ----------
active_char = None
GM_input = ""
console = Console()

# ---------- Command Processing ----------
class GMInterface:
    def __init__(self, agent: OllamaAgent, prompt_manager: PromptManager,
                current_submode: str = "roleplay"):
        
        self.agent = agent
        self.pm = prompt_manager
        self.current_submode = current_submode
        self.submode_instruction_text = self.pm.submode(current_submode)
        self.pm.scene_raw = self.agent.read_active_scene()
        # Cached base prompts
        self.SYSTEM_PROMPT = self.pm.system_prompt()
        self.CHARACTER_INSTRUCTIONS = self.pm.character_instructions()
        self.batcher = BatchManager(self.agent)
        self.retry_feedback = []
        self.auto_mode = False

    # Existing methods like show_help, normalize_llm_output, list_characters, next_character, etc.

    # ------------------- NEW HELPER -------------------
    def _send_to_llm(self, user_input: str):
        """
        Send GM input to the LLM, using a collapsed scene (summaries for old turns)
        instead of the full scene, append output to scene, store messages for retry.
        """
        # --- Append GM input first ---
        if user_input.strip():
            self.agent.append_to_active_scene(f"GM : {user_input}")

        # --- Build collapsed scene for LLM ---
        collapsed_scene = self.pm.build_scene_text(turns_to_keep=None)

        speaker_name = self.agent.character_names[self.agent.active_character_index]
        active_char_sheet_text = ""
        # --- Build messages based on submode ---
        if self.current_submode == "group":
                # New group messages builder handles scene and sheets internally
                messages = self.pm.build_group_messages(
                    self.agent,
                    self.SYSTEM_PROMPT,
                    collapsed_scene,
                    user_input
                )
                speaker_name = "Group"

        else:
            # Single-character mode
            active_char_path = self.agent.character_paths[self.agent.active_character_index]
            active_char_sheet_text = read_vault_file(
                self.agent.vault_root,
                str(active_char_path.relative_to(self.agent.vault_root))
            )

            messages = self.pm.build_single_character_messages(
                system_prompt=self.SYSTEM_PROMPT,
                character_instructions=self.CHARACTER_INSTRUCTIONS,
                submode_instructions=self.submode_instruction_text,
                character_sheet=active_char_sheet_text,
                scene_text=collapsed_scene,  # <<< use collapsed scene
                user_input=user_input,
                speaker_name=speaker_name,
            )

        # --- Count tokens ---
        tokens_used, breakdown = self.agent.count_tokens(
            messages, model_to_use=self.agent.model, return_breakdown=True
        )
        console.print(f"[Token Usage] {tokens_used} tokens: {breakdown}")
        self.agent._last_token_usage = tokens_used

        # --- Check against model limit ---
        from config import DEFAULT_MODEL_TOKEN_LIMIT, check_context_usage
        check_context_usage(tokens_used, DEFAULT_MODEL_TOKEN_LIMIT)

        # --- Store for retry ---
        self.agent._retry_context = {
            "retry_system_prompt": self.SYSTEM_PROMPT,
            "retry_character_instructions": self.CHARACTER_INSTRUCTIONS,
            "retry_submode_instructions": self.submode_instruction_text,
            "retry_character_sheet": active_char_sheet_text,   # as you want
            "retry_speaker_name": speaker_name,
            "retry_scene_text_snapshot": collapsed_scene,  # correct for your workflow
            "retry_user_input": user_input,
}
        # --- Call LLM ---
        response = self.agent.chat(messages)

        # --- Normalize & append ---
        char_response = self.normalize_llm_output(response, speaker_name)
        console.print(char_response)
        self.agent.append_llm_output(char_response)



    # ------------------- AUTO NEXT -------------------
    def auto_next_character(self):
        self.next_character()
        self._send_to_llm("")

    def show_help(self):
        console.print(HELP_LINES)
        return HELP_LINES

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
        for i in range(1, len(name_parts) + 1):
            sub = " ".join(name_parts[:i])  # "Professor", then "Professor Whizzlebum"
            patterns.append(re.escape(sub))
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


    def summarize_scene(self, turns_to_keep: int = None):
        """
        Manual trigger for auto-summarizing old turns.
        Optional: specify number of recent turns to keep unsummarized.
        """
        abs_scene_path = self.agent.get_active_scene_path()

        if not abs_scene_path or not abs_scene_path.exists():
            console.print("[red]No active scene file to summarize.[/red]")
            return

        console.print("[cyan]Manually summarizing old turns…[/cyan]")

        # Trigger the turn-by-turn summarizer
        summarize_scene_turns(abs_scene_path, self.agent, turns_to_keep=turns_to_keep)

        console.print("[bold green]Turn summaries updated.[/bold green]")

    def regenerate_last(self):
        # Retrieve the parameters used to create the last message batch
        ctx = getattr(self.agent, "_retry_context", None)
        if not ctx:
            console.print("[yellow]Nothing to retry.[/yellow]")
            return

        # Unpack stored values (already shortened)
        system_prompt          = ctx["retry_system_prompt"]
        character_instructions = ctx["retry_character_instructions"]
        submode_instructions   = ctx["retry_submode_instructions"]
        character_sheet        = ctx["retry_character_sheet"]
        speaker_name           = ctx["retry_speaker_name"]
        collapsed_scene        = ctx["retry_scene_text_snapshot"]  # already collapsed
        user_input             = ctx["retry_user_input"]

        # Roll back previous LLM output
        rolled = self.agent.rollback_last_llm_output()
        if not rolled:
            console.print("[yellow]No previous LLM output to roll back.[/yellow]")

        # ---- Inject retry feedback BEFORE rebuilding messages ----
        if self.retry_feedback:
            feedback = "\n".join(self.retry_feedback)
            user_input = user_input + "\n" + feedback
            ctx["retry_user_input"] = user_input  # update stored version

        # ---- Rebuild message stack ----
    # ---- Rebuild message stack depending on mode ----
        if self.current_submode == "group":
            # Use group method
            messages = self.pm.build_group_messages(
                    agent=self.agent,
                    system_prompt=system_prompt,
                    scene_text=collapsed_scene,   # already collapsed
                    user_input=user_input
                )
            speaker_name = "Group"
        else:
            # Single-character method
            messages = self.pm.build_single_character_messages(
                system_prompt=system_prompt,
                character_instructions=character_instructions,
                submode_instructions=submode_instructions,
                character_sheet=character_sheet,
                speaker_name=speaker_name,
                scene_text=collapsed_scene,   # <-- already collapsed
                user_input=user_input         # <-- now includes feedback
            )

        console.print("[cyan]Regenerating last LLM response…[/cyan]")
        response = self.agent.chat(messages)

        # ---- Normalize and append to scene ----
        speaker_name = self.agent.character_names[self.agent.active_character_index]
        char_response = self.normalize_llm_output(response, speaker_name)
        self.agent.append_llm_output(char_response)

        # ---- Print updated output ----
        console.print("\n[bold green]Updated Response:[/bold green]")
        console.print(char_response + "\n")

    def summarize_full_scene(self, scene_text: str) -> str:
        console.print("[cyan]Summarizing full scene…[/cyan]")

        summary_batches = self.batcher.get_tokenwise_summary_batches(
            scene_text=scene_text,
            system_prompts=[{"role": "system", "content": self.pm.summary_prompt()}],
            SCENE_CONTEXT_THRESHOLD=SCENE_CONTEXT_THRESHOLD,
            prompt_manager=self.pm,
            model_token_limit=DEFAULT_MODEL_TOKEN_LIMIT,
        )

        accumulated_summary = ""
        total = len(summary_batches)
        console.print(f"[cyan]Created {total} summarization batch(es).[/cyan]")

        for i, batch in enumerate(summary_batches, start=1):
            messages = self.pm.build_summary_messages(
                scene_text=batch['batch_text'],
                prior_summary_text=batch['prior_summary_text']  # empty for first batch
            )

            used_tokens = self.agent.count_tokens(messages)
            console.print(f"\n[bold cyan]Processing batch {i}/{total}…[/bold cyan]")
            console.print(f"[magenta]Batch {i} token usage: {used_tokens} tokens[/magenta]")
            console.print(f"[magenta]Turns in this batch: {batch['turn_indices']}[/magenta]")
            console.print(f"[yellow]Requesting LLM summary for batch {i}…[/yellow]")

            llm_output = self.agent.chat(messages).strip()
            console.print(f"[green]Received summary for batch {i}.[/green]")

            accumulated_summary += ("\n\n" if accumulated_summary else "") + llm_output
            batch["prior_summary_text"] = accumulated_summary

        console.print("\n[bold green]All batches processed![/bold green]")
        return accumulated_summary.strip()
    
    # ---------- MAIN INTERACTIVE LOOP ----------
    def run(self):
        console.print("[bold cyan]GM Assistant Ready.[/bold cyan]\n")

        while True:
            GM_input = input(f"\n({self.current_submode}) {self.agent.character_names[self.agent.active_character_index]} GM> ").strip()
            scene_path = self.agent.get_active_scene_path()
            current_turn = ensure_current_turn(scene_path)

            # -----------------------------------------------------
            # 1) Handle TRY AGAIN and /p BEFORE any other commands
            # -----------------------------------------------------
            if GM_input.lower().startswith("try again") or GM_input.startswith("/p"):
                parts = GM_input.split(" ", 1)
                feedback = parts[1].strip() if len(parts) > 1 else ""

                if feedback:
                    self.retry_feedback.append(feedback)

                self.regenerate_last()
                continue

            # Narration-only append
            if GM_input.startswith("."):
                narration_text = "GM : " + GM_input[1:].lstrip()
                self.agent.append_to_active_scene(narration_text)
                console.print(f"[Narration appended]", style="bold cyan")
                continue  # skip LLM response

            # ------------------ Toggle auto mode ------------------
            if GM_input == "*":
                self.auto_mode = not self.auto_mode
                status = "activated" if self.auto_mode else "deactivated"
                console.print(f"[bold cyan]Auto mode {status}[/bold cyan]")
                continue

            if GM_input.startswith("/"):
                # Help
                if GM_input == "/h":
                    self.show_help()
                    continue

                # Roll dice
                elif GM_input.startswith("/r "):
                    self.handle_roll(GM_input[3:].strip())
                    continue

                # Summarize scene, optionally with turns_to_keep
                elif GM_input.startswith("/s"):
                    parts = GM_input.split(maxsplit=1)
                    turns_to_keep = None

                    if len(parts) > 1 and parts[1].isdigit():
                        turns_to_keep = int(parts[1])

                    self.summarize_scene(turns_to_keep=turns_to_keep)
                    continue


                # List characters
                elif GM_input == "/ls":
                    self.list_characters()
                    continue

                # Next character
                elif GM_input == "/n":
                    self.next_character()
                    continue

                # Next turn
                elif GM_input == "/t":
                    scene_path = self.agent.get_active_scene_path()
                    advance_turn(scene_path, self.agent, current_turn=current_turn)
                    continue

                # Submode shortcuts (/c, /r, /e)
                elif GM_input.lower() in ("/c", "/r", "/g", "/e"):
                    if GM_input.lower() == "/c":
                        self.current_submode = "combat"
                    elif GM_input.lower() == "/r":
                        self.current_submode = "roleplay"
                    elif GM_input.lower() == "/g":
                        self.current_submode = "group"
                        self.submode_instruction_text = ""
                        console.print("[bold cyan]Submode switched to group[/bold cyan]")
                        continue
                    else:
                        self.current_submode = "exploration"
                        self.submode_instruction_text = self.pm.submode(self.current_submode)
                        console.print(f"[bold cyan]Submode switched to {self.current_submode}[/bold cyan]")
                        continue

                # Switch active character by number (/1 /2 /3)
                elif GM_input.startswith("/") and GM_input[1:].isdigit():
                    self.switch_character(int(GM_input[1:]))
                    continue

                elif GM_input == "/end":

                    # Get only the summary string (summarize_full_scene performs LLM calls)
                    final_summary = self.summarize_full_scene(self.pm.scene_raw)

                    if not final_summary:
                        console.print("[yellow]No summary returned from summarizer.[/yellow]")
                        continue

                    # Read original scene
                    original = self.agent.read_active_scene()

                    # Remove any existing Scene Summary blocks (keep rest)
                    # Pattern: header "# Scene Summary" (or "## Scene Summary") and any following lines
                    import re
                    # Remove any existing Scene Summary section(s)
                    cleaned = re.sub(
                        r"(?ms)^\s*#{1,2}\s*Scene Summary\s*\n.*?(?=^\s*#\s*(Turn\s+\d+|Scene Summary)\b|\Z)",
                        "",
                        original
                    ).rstrip()

                    # Detect description vs start at Turn 1
                    m = re.search(r"^#\s*Turn\s+1\b", cleaned, flags=re.M)
                    if m:
                        desc_block = cleaned[:m.start()].strip()
                        turns_block = cleaned[m.start():].lstrip()
                    else:
                        # No Turn 1 found — treat whole file as turns_block fallback
                        desc_block = ""
                        turns_block = cleaned.strip()

                    # Build the new scene text: description (if any), Scene Summary, then turns
                    parts = []
                    if desc_block:
                        parts.append(desc_block)
                    parts.append("# Scene Summary\n" + final_summary.strip())
                    if turns_block:
                        parts.append(turns_block)

                    new_scene = "\n\n".join(parts).strip() + "\n"

                    # Write back the scene file (replace)
                    # Use agent or helper that writes whole scene — replace with your write function
                    try:
                        # if your agent has a write_scene or similar, use it. Otherwise overwrite file.
                        scene_path = self.agent.get_active_scene_path()
                        scene_path.write_text(new_scene, encoding="utf-8")
                        # Refresh pm.scene_text and agent internal state if needed
                        self.pm.scene_raw = new_scene
                        console.print("\n[bold green]Full scene summary written into scene file.[/bold green]")
                        console.print("# Scene Summary\n" + final_summary.strip())
                    except Exception as e:
                        console.print(f"[red]Failed to write scene file: {e}[/red]")
                        console.print("# Scene Summary\n" + final_summary.strip())

                    continue

                # Unknown command
                else:
                    console.print("[yellow]Unknown command.[/yellow]")
                    continue

            # Empty input → treat as "GM says nothing" and continue the scene
            if not GM_input:
                if getattr(self, "auto_mode", False):
                    self.auto_next_character()   # switches character AND sends empty input
                else:
                    self._send_to_llm("")        # just sends empty input
                continue


                # NORMAL FLOW: GM provides input → LLM responds → append to scene

            # Use the helper to send GM input to LLM and append
            self._send_to_llm(GM_input)

    # ---------- Main ----------
def main():
    agent = OllamaAgent(vault_root, characters_dir, scenes_active_dir)
    scene_text = agent.read_active_scene()
    # Create PromptManager
    pm = PromptManager(prompts_dir)

    

    gm = GMInterface(
        agent=agent,
        prompt_manager=pm,
        current_submode="roleplay",
    )

    gm.run()

if __name__ == "__main__":
    main()
