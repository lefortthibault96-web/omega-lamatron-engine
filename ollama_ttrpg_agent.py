#!/usr/bin/env python3
import re
from pathlib import Path
from rich.console import Console
from snitch import SnitchEditor, write_vault_file, run_snitch_auto_detection
from dice import roll_dice
from LLM import OllamaAgent
from Prompt_Manager2000 import PromptManager
from config import vault_root, characters_dir, scenes_active_dir, prompts_dir, HELP_LINES, DYNAMIC_SUMMARY_ALLOW_MIDTURN
from turns import ensure_current_turn, advance_turn, summarize_scene_turns, summarize_turn_fulltext_batches, renumber_turns, dynamic_summarize_scene
from batch import BatchManager
from utils import read_vault_file, DEFAULT_MODEL_TOKEN_LIMIT, check_context_usage, remove_empty_turns
from commands import build_default_registry, CommandParser

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
        # Cached base prompts
        self.SYSTEM_PROMPT = self.pm.system_prompt()
        self.CHARACTER_INSTRUCTIONS = self.pm.character_instructions()
        self.batcher = BatchManager(self.agent)
        self.retry_feedback = []
        self.auto_mode = False

        # Expose console on the instance so commands.py can use gm.console
        self.console = console

        # Command system
        self.registry = build_default_registry()
        self.parser = CommandParser(self.registry)

    # ------------------- CORE LLM HELPER -------------------
    def _send_to_llm(self, user_input: str):
        """
        Send GM input to the LLM, using a dynamically collapsed scene (summaries for old turns)
        instead of the full scene, append output to scene, store messages for retry.
        """

        # --- Append GM input first ---
        if user_input.strip():
            self.agent.append_to_active_scene(f"GM : {user_input}")

        # ======================================================
        # Build dynamically summarized scene for LLM
        # ======================================================
        candidate_text, included_summaries, included_fulls = dynamic_summarize_scene(
            pm=self.pm,
            batcher=self.batcher,
            agent=self.agent,
            allow_fulltext_collapse=None   # obey config flag
        )

        collapsed_scene = candidate_text

        speaker_name = self.agent.character_names[self.agent.active_character_index]
        active_char_sheet_text = ""

        # --- Build messages based on submode ---
        if self.current_submode == "group":

            messages = self.pm.build_group_messages(
                self.agent,
                self.SYSTEM_PROMPT,
                collapsed_scene,
                user_input
            )
            speaker_name = "Group"

        else:
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
                scene_text=collapsed_scene,
                user_input=user_input,
                speaker_name=speaker_name,
            )

        # --- Count tokens ---
        tokens_used, breakdown = self.agent.count_tokens(
            messages, model_to_use=self.agent.model, return_breakdown=True
        )
        console.print(f"[Token Usage] {tokens_used} tokens: {breakdown}")
        self.agent._last_token_usage = tokens_used

        # --- Check context limit ---
        check_context_usage(tokens_used, DEFAULT_MODEL_TOKEN_LIMIT)

        # --- Store for retry ---
        self.agent._retry_context = {
            "retry_system_prompt": self.SYSTEM_PROMPT,
            "retry_character_instructions": self.CHARACTER_INSTRUCTIONS,
            "retry_submode_instructions": self.submode_instruction_text,
            "retry_character_sheet": active_char_sheet_text,
            "retry_speaker_name": speaker_name,
            "retry_scene_text_snapshot": collapsed_scene,
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
        patterns = []
        for i in range(1, len(name_parts) + 1):
            sub = " ".join(name_parts[:i])
            patterns.append(re.escape(sub))
        pattern = rf"^(?:{'|'.join(patterns)})\s*:"

        # Remove the prefix if it exists
        response = re.sub(pattern, '', response, count=1).lstrip()

        # Prepend canonical speaker
        return f"{speaker_name} : {response}"

    # --- Character helpers ---
    def list_characters(self):
        console.print("\n[bold cyan]Characters:[/bold cyan]")
        for i, name in enumerate(self.agent.character_names):
            marker = "[active]" if i == self.agent.active_character_index else ""
            console.print(f"{i+1}. {name} {marker}")
        console.print("")

    def switch_character(self, idx: int):
        if 1 <= idx <= len(self.agent.character_names):
            self.agent.active_character_index = idx - 1
            console.print(f"[green]Active character is now: {self.agent.character_names[idx-1]}[/green]")
        else:
            console.print("[red]Invalid character index[/red]")

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
        except Exception:
            console.print("[red]Invalid dice expression[/red]")

    # --- Regenerate last LLM response ---
    def regenerate_last(self):
        ctx = getattr(self.agent, "_retry_context", None)
        if not ctx:
            console.print("[yellow]Nothing to retry.[/yellow]")
            return

        system_prompt          = ctx["retry_system_prompt"]
        character_instructions = ctx["retry_character_instructions"]
        submode_instructions   = ctx["retry_submode_instructions"]
        character_sheet        = ctx["retry_character_sheet"]
        speaker_name           = ctx["retry_speaker_name"]
        collapsed_scene        = ctx["retry_scene_text_snapshot"]
        user_input             = ctx["retry_user_input"]

        rolled = self.agent.rollback_last_llm_output()
        if not rolled:
            console.print("[yellow]No previous LLM output to roll back.[/yellow]")

        # Inject retry feedback
        if self.retry_feedback:
            feedback = "\n".join(self.retry_feedback)
            user_input = user_input + "\n" + feedback
            ctx["retry_user_input"] = user_input

        # Rebuild message stack
        if self.current_submode == "group":
            messages = self.pm.build_group_messages(
                agent=self.agent,
                system_prompt=system_prompt,
                scene_text=collapsed_scene,
                user_input=user_input
            )
            speaker_name = "Group"
        else:
            messages = self.pm.build_single_character_messages(
                system_prompt=system_prompt,
                character_instructions=character_instructions,
                submode_instructions=submode_instructions,
                character_sheet=character_sheet,
                speaker_name=speaker_name,
                scene_text=collapsed_scene,
                user_input=user_input
            )

        console.print("[cyan]Regenerating last LLM response…[/cyan]")
        response = self.agent.chat(messages)

        speaker_name = self.agent.character_names[self.agent.active_character_index]
        char_response = self.normalize_llm_output(response, speaker_name)
        self.agent.append_llm_output(char_response)

        console.print("\n[bold green]Updated Response:[/bold green]")
        console.print(char_response + "\n")

    # ============================================================
    # COMMAND HANDLERS (used by commands.py)
    # ============================================================

    def cmd_force_summarize(self, arg: str):
        """
        Force a dynamic summarization pass.
        Optional arg = number of full turns to keep (currently informational).
        """
        scene_path = self.agent.get_active_scene_path()
        scene_text = self.agent.read_active_scene()

        # 1. Remove empty turns
        lines = scene_text.splitlines()
        lines = remove_empty_turns(lines)
        scene_text = "\n".join(lines).strip() + "\n"
        scene_path.write_text(scene_text, encoding="utf-8")

        # 2. Renumber
        renumber_turns(scene_path)

        # 3. Summarize old turns
        summarize_scene_turns(self.agent, None)

        # 4. Optional info about turns_to_keep
        if arg.isdigit():
            turns_to_keep = int(arg)
            console.print(f"[cyan]Forcing collapse while keeping last {turns_to_keep} turns (config-dependent).[/cyan]")
        else:
            console.print("[cyan]Forcing collapse (mid-turn allowed)…[/cyan]")

        # 5. Hard collapse pass
        summarize_turn_fulltext_batches(
            self.pm,
            self.batcher,
            self.agent,
            scene_text=self.agent.read_active_scene(),
            max_batches=1,
            mode="dynamic"
        )

        # 6. Rebuild dynamic batch with full collapse allowed
        candidate_text, included_summaries, included_fulls = dynamic_summarize_scene(
            pm=self.pm,
            batcher=self.batcher,
            agent=self.agent,
            allow_fulltext_collapse=True
        )

        console.print("[green]Scene collapsed.[/green]")
        console.print(
            f"[cyan]Dynamic batch includes: "
            f"{len(included_summaries)} summaries, "
            f"{len(included_fulls)} full turns.[/cyan]"
        )

    def cmd_next_turn(self):
        scene_path = self.agent.get_active_scene_path()
        scene_text = self.agent.read_active_scene()

        # 1. Remove empty shells
        lines = remove_empty_turns(scene_text.splitlines())
        scene_text = "\n".join(lines).strip() + "\n"
        scene_path.write_text(scene_text, encoding="utf-8")

        # 2. Renumber sequentially
        renumber_turns(scene_path)

        # 3. Summarize old turns
        summarize_scene_turns(self.agent, None)

        # 4. Determine current turn
        current_turn = ensure_current_turn(scene_path)

        # 5. Open NEW turn as "(In Progress)"
        advance_turn(scene_path, self.agent, current_turn=current_turn)

        # 6. Dynamic summarization after advancing
        candidate_text, included_summaries, included_fulls = dynamic_summarize_scene(
            pm=self.pm,
            batcher=self.batcher,
            agent=self.agent,
            allow_fulltext_collapse=True
        )

        console.print(
            f"[cyan]Dynamic batch built after advancing from Turn {current_turn}:[/cyan]"
        )
        console.print(
            f"[cyan]{len(included_summaries)} summaries, {len(included_fulls)} full turns included.[/cyan]"
        )

    def cmd_end_scene(self):
        scene_path = self.agent.get_active_scene_path()
        scene_text = self.agent.read_active_scene()

        # 1. Close any (In Progress) turn if your agent supports it
        if hasattr(self.agent, "close_turn_in_progress"):
            self.agent.close_turn_in_progress()
            scene_text = self.agent.read_active_scene()

        # 2. Remove empty turns
        lines = remove_empty_turns(scene_text.splitlines())
        scene_text = "\n".join(lines).strip() + "\n"
        scene_path.write_text(scene_text, encoding="utf-8")

        # 3. Renumber
        renumber_turns(scene_path)

        # 4. Full-scene collapse (mode="end" collapses all remaining turns)
        summarize_turn_fulltext_batches(
            self.pm,
            self.batcher,
            self.agent,
            scene_text=self.agent.read_active_scene(),
            mode="end"
        )

        console.print("\n[bold green]Full scene summary written into scene file.[/bold green]")

    # ---------- MAIN INTERACTIVE LOOP ----------
    def run(self):
        console.print("[bold cyan]GM Assistant Ready.[/bold cyan]\n")

        while True:
            GM_input = input(
                f"\n({self.current_submode}) "
                f"{self.agent.character_names[self.agent.active_character_index]} GM> "
            ).strip()

            scene_path = self.agent.get_active_scene_path()
            current_turn = ensure_current_turn(scene_path)

            # --------------------------------------------------------
            # 0) Narration-only append (starts with ".")
            # --------------------------------------------------------
            if GM_input.startswith("."):
                narration_text = "GM : " + GM_input[1:].lstrip()
                self.agent.append_to_active_scene(narration_text)
                console.print("[Narration appended]", style="bold cyan")
                continue

            # --------------------------------------------------------
            # 1) Empty input → auto mode OR send empty input to LLM
            # --------------------------------------------------------
            if not GM_input:
                if self.auto_mode:
                    self.auto_next_character()
                else:
                    self._send_to_llm("")
                continue

            # --------------------------------------------------------
            # 2) ALL commands (including "*", try again, /p, /x)
            # --------------------------------------------------------
            handler, arg = self.parser.parse(GM_input)
            if handler:
                handler.execute(self, arg)
                continue

            # --------------------------------------------------------
            # 3) NORMAL FLOW → send GM input to LLM
            # --------------------------------------------------------
            self._send_to_llm(GM_input)

    # ---------- Main ----------
def main():
    agent = OllamaAgent(vault_root, characters_dir, scenes_active_dir)
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
