# commands.py

import re


# ============================================================
# BASE CLASS FOR COMMANDS
# ============================================================

class GMCommand:
    """
    Base class for all GM commands.
    Each command implements:
        - name ("/cmd")
        - optional aliases
        - execute(gm, arg)
    """
    name = ""
    aliases = []

    def execute(self, gm, arg: str):
        raise NotImplementedError


# ============================================================
# COMMAND REGISTRY
# ============================================================

class CommandRegistry:
    def __init__(self):
        self.commands = {}

    def register(self, command: GMCommand):
        self.commands[command.name] = command
        for alias in command.aliases:
            self.commands[alias] = command

    def get(self, name: str):
        return self.commands.get(name)


# ============================================================
# COMMAND PARSER
# ============================================================

class CommandParser:
    def __init__(self, registry: CommandRegistry):
        self.registry = registry

    def parse(self, text: str):
        """
        Returns: (command, arg) or (None, None)
        """
        # Special non-slash command: "*" toggles auto mode
        if text == "*":
            cmd = self.registry.get("*")
            return cmd, ""
        
        # Special non-slash retry command: "try again ..." or "/p ..."
        lower = text.lower()
        if lower.startswith("try again"):
            cmd = self.registry.get("try-again")
            arg = text[10:].strip()  # everything after "try again"
            return cmd, arg

        if not text.startswith("/"):
            return None, None

        # Character switch: /1 /2 /3...
        if text[1:].isdigit():
            cmd = self.registry.get("/digit")
            return cmd, text[1:]
        
        if lower in ("/c", "/r", "/g", "/e"):
            cmd = self.registry.get("/submode")
            return cmd, lower

        # Dice roll: /r X
        if text.startswith("/r "):
            cmd = self.registry.get("/r")
            return cmd, text[3:].strip()

        # Split single-arg commands: /s, /s 5
        parts = text.split(" ", 1)
        cmd_name = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        cmd = self.registry.get(cmd_name)
        if cmd:
            return cmd, arg

        # Unknown
        return self.registry.get("/unknown"), text


# ============================================================
# INDIVIDUAL COMMANDS
# ============================================================

class HelpCommand(GMCommand):
    name = "/h"
    aliases = ["/help", "/?"]

    def execute(self, gm, arg):
        gm.show_help()


class RollCommand(GMCommand):
    name = "/r"

    def execute(self, gm, expr):
        if not expr:
            gm.console.print("[red]Usage: /r 2d6+3[/red]")
            return
        gm.handle_roll(expr)


class NextCharacterCommand(GMCommand):
    name = "/n"

    def execute(self, gm, arg):
        gm.next_character()


class ListCharactersCommand(GMCommand):
    name = "/ls"

    def execute(self, gm, arg):
        gm.list_characters()


class SwitchCharacterCommand(GMCommand):
    name = "/digit"   # Special marker used by parser

    def execute(self, gm, idx_str):
        gm.switch_character(int(idx_str))


class ToggleAutoModeCommand(GMCommand):
    name = "*"

    def execute(self, gm, arg):
        gm.auto_mode = not gm.auto_mode
        gm.console.print(
            f"[cyan]Auto mode {'activated' if gm.auto_mode else 'deactivated'}[/cyan]"
        )


class SummarizeCommand(GMCommand):
    name = "/s"

    def execute(self, gm, arg):
        """
        /s           → force collapse
        /s 5         → force collapse keeping last 5 turns
        """
        gm.cmd_force_summarize(arg)   # We'll create cmd_force_summarize inside GMInterface


class NextTurnCommand(GMCommand):
    name = "/t"

    def execute(self, gm, arg):
        gm.cmd_next_turn()   # GMInterface method


class EndSceneCommand(GMCommand):
    name = "/end"

    def execute(self, gm, arg):
        gm.cmd_end_scene()   # GMInterface method


class UnknownCommand(GMCommand):
    name = "/unknown"

    def execute(self, gm, text):
        gm.console.print(f"[yellow]Unknown command: {text}[/yellow]")

class RetryCommand(GMCommand):
    name = "try-again"    # internal name, not user-visible

    def execute(self, gm, feedback):
        if feedback:
            gm.retry_feedback.append(feedback)
        gm.regenerate_last()

class ScrunchCommand(GMCommand):
    name = "/scrunch"

    def execute(self, gm, arg):
        from turns import scrunch
        scrunch(gm.agent, gm.pm)

class SubmodeCommand(GMCommand):
    name = "/submode"   # internal router key

    def execute(self, gm, arg):
        mapping = {
            "/c": "combat",
            "/r": "roleplay",
            "/g": "group",
            "/e": "exploration",
        }
        mode = mapping.get(arg.lower())
        if mode:
            gm.set_submode(mode)
        else:
            gm.console.print(f"[red]Unknown submode: {arg}[/red]")


# ============================================================
# REGISTER ALL COMMANDS FOR GMInterface
# ============================================================

def build_default_registry():
    registry = CommandRegistry()

    registry.register(HelpCommand())
    registry.register(RollCommand())
    registry.register(NextCharacterCommand())
    registry.register(ListCharactersCommand())
    registry.register(SwitchCharacterCommand())
    registry.register(ToggleAutoModeCommand())
    registry.register(SummarizeCommand())
    registry.register(NextTurnCommand())
    registry.register(EndSceneCommand())
    registry.register(UnknownCommand())
    registry.register(RetryCommand())
    registry.register(ScrunchCommand())
    registry.register(SubmodeCommand())

    return registry
