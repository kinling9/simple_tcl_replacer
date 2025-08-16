#!/usr/bin/env python3
"""
TCL Script Translator
Supports translating TCL commands based on configuration files, handles nested calls and error reporting
"""

import re
import toml
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TranslationRule:
    """Translation rule data class"""

    original_cmd: str
    target_cmd: str
    param_mappings: Dict[str, str] = (
        None  # Parameter mappings, e.g. {"-name": "-by_name"}
    )

    def __post_init__(self):
        if self.param_mappings is None:
            self.param_mappings = {}


class TCLTranslator:
    """TCL script translator main class"""

    def __init__(self, config_file: str):
        self.translation_rules: Dict[str, TranslationRule] = {}
        self.native_tcl_commands = {
            "set",
            "puts",
            "if",
            "else",
            "elseif",
            "while",
            "for",
            "foreach",
            "proc",
            "return",
            "break",
            "continue",
            "switch",
            "case",
            "default",
            "list",
            "lappend",
            "lindex",
            "llength",
            "lrange",
            "lsearch",
            "lsort",
            "string",
            "regexp",
            "regsub",
            "split",
            "join",
            "format",
            "scan",
            "array",
            "dict",
            "info",
            "namespace",
            "variable",
            "global",
            "upvar",
            "source",
            "load",
            "package",
            "catch",
            "error",
            "throw",
            "try",
            "open",
            "close",
            "read",
            "write",
            "gets",
            "puts",
            "flush",
            "seek",
            "tell",
            "eof",
            "file",
            "glob",
            "pwd",
            "cd",
            "exec",
            "exit",
            "expr",
            "incr",
            "append",
            "unset",
            "eval",
            "subst",
            "after",
        }
        self.load_config(config_file)

    def load_config(self, config_file: str):
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"config_file cannot be found: {config_file}")

        self._load_toml_config(config_path)

    def _load_toml_config(self, config_path: Path):
        """Load TOML configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        self._parse_config_data(config)

    def _parse_config_data(self, config: Dict[str, Any]):
        """Parse configuration data"""
        translations = config.get("translations", [])

        for rule_data in translations:
            original_cmd = rule_data["original_cmd"]
            target_cmd = rule_data["target_cmd"]
            param_mappings = rule_data.get("param_mappings", {})

            rule = TranslationRule(
                original_cmd=original_cmd,
                target_cmd=target_cmd,
                param_mappings=param_mappings,
            )

            self.translation_rules[original_cmd] = rule

    def parse_tcl_command(self, line: str) -> List[Tuple[str, List[str]]]:
        """
        Parse TCL command line, return command and parameter list
        Handle nested bracket calls
        """
        commands = []
        i = 0

        while i < len(line):
            # Skip whitespace characters
            while i < len(line) and line[i].isspace():
                i += 1

            if i >= len(line):
                break

            # Find start of command
            if line[i] == "[":
                # Handle nested commands
                bracket_count = 1
                start = i + 1
                i += 1

                while i < len(line) and bracket_count > 0:
                    if line[i] == "[":
                        bracket_count += 1
                    elif line[i] == "]":
                        bracket_count -= 1
                    i += 1

                if bracket_count == 0:
                    nested_cmd = line[start : i - 1]
                    nested_commands = self.parse_tcl_command(nested_cmd)
                    commands.extend(nested_commands)
                else:
                    # Unclosed brackets
                    break
            else:
                # Handle normal commands
                start = i
                while i < len(line) and line[i] not in "[]":
                    i += 1

                cmd_line = line[start:i].strip()
                if cmd_line:
                    parsed = self._parse_single_command(cmd_line)
                    if parsed:
                        commands.append(parsed)

        return commands

    def _parse_single_command(self, cmd_line: str) -> Optional[Tuple[str, List[str]]]:
        """Parse a single command line"""
        if not cmd_line.strip():
            return None

        # Use regular expressions to parse commands and parameters, considering quotes
        tokens = []
        current_token = ""
        in_quotes = False
        quote_char = None
        i = 0

        while i < len(cmd_line):
            char = cmd_line[i]

            if not in_quotes:
                if char in ['"', "'"]:
                    in_quotes = True
                    quote_char = char
                    current_token += char
                elif char.isspace():
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                else:
                    current_token += char
            else:
                current_token += char
                if char == quote_char:
                    in_quotes = False
                    quote_char = None

            i += 1

        if current_token:
            tokens.append(current_token)

        if not tokens:
            return None

        command = tokens[0]
        args = tokens[1:] if len(tokens) > 1 else []

        return command, args

    def translate_command(self, command: str, args: List[str]) -> Tuple[str, List[str]]:
        """Translate a single command"""
        # Check if it's a native TCL command
        if command in self.native_tcl_commands:
            return command, args

        # Check if there's a translation rule
        if command not in self.translation_rules:
            return (
                command,
                args,
            )  # Return original command, let caller decide whether to warn

        rule = self.translation_rules[command]
        new_command = rule.target_cmd
        new_args = []

        # Translate parameters
        i = 0
        while i < len(args):
            arg = args[i]

            # Check if it's a parameter flag (starts with -)
            if arg.startswith("-") and arg in rule.param_mappings:
                new_args.append(rule.param_mappings[arg])
                # If there's a parameter value, add it too
                if i + 1 < len(args):
                    i += 1
                    new_args.append(args[i])
            else:
                new_args.append(arg)

            i += 1

        return new_command, new_args

    def translate_line(self, line: str, line_num: int) -> Tuple[str, List[str]]:
        """
        Translate a line of TCL code
        Return translated line and warning list
        """
        warnings = []

        # Parse commands
        commands = self.parse_tcl_command(line.strip())

        if not commands:
            return line, warnings

        translated_parts = []
        original_line = line

        for command, args in commands:
            # Translate command
            if command in self.native_tcl_commands:
                # Native command, no translation
                translated_cmd = command
                translated_args = args
            elif command in self.translation_rules:
                # Has translation rule
                translated_cmd, translated_args = self.translate_command(command, args)
            else:
                # Non-native command without translation rule
                warnings.append(f"line: {line_num}: unsupport command: '{command}'")
                translated_cmd = command
                translated_args = args

            # Reconstruct command string
            if translated_args:
                cmd_str = f"{translated_cmd} {' '.join(translated_args)}"
            else:
                cmd_str = translated_cmd

            translated_parts.append(cmd_str)

        # Reconstruct entire line, maintaining original nested structure
        if len(commands) == 1:
            # Single command
            result = translated_parts[0]
        else:
            # Multiple commands, need to maintain nested structure
            result = self._reconstruct_nested_line(
                original_line, commands, translated_parts
            )

        return result, warnings

    def _reconstruct_nested_line(
        self,
        original_line: str,
        parsed_commands: List[Tuple[str, List[str]]],
        translated_parts: List[str],
    ) -> str:
        """Reconstruct line containing nested commands"""
        # Simplified processing: for nested commands, directly replace each recognized command
        result = original_line

        for i, (original_cmd, original_args) in enumerate(parsed_commands):
            if i < len(translated_parts):
                # Build original command string
                original_cmd_str = (
                    f"{original_cmd} {' '.join(original_args)}"
                    if original_args
                    else original_cmd
                )

                # Replace in result
                result = result.replace(
                    f"[{original_cmd_str}]", f"[{translated_parts[i]}]"
                )
                result = result.replace(original_cmd_str, translated_parts[i])

        return result

    def translate_file(self, input_file: str, output_file: str = None) -> List[str]:
        """Translate entire TCL file"""
        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"input_file {input_file} not exists!")

        if output_file is None:
            output_file = str(input_path.with_suffix(".translated" + input_path.suffix))

        all_warnings = []
        translated_lines = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                translated_line, warnings = self.translate_line(
                    line.rstrip("\n"), line_num
                )
                translated_lines.append(translated_line)
                all_warnings.extend(warnings)

        # Write translated file
        with open(output_file, "w", encoding="utf-8") as f:
            for line in translated_lines:
                f.write(line + "\n")

        return all_warnings


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TCL Script Translator")
    parser.add_argument("--config", "-c", required=True, help="Configuration file path")
    parser.add_argument("--input", "-i", required=True, help="Input TCL file")
    parser.add_argument("--output", "-o", help="Output file path (optional)")

    args = parser.parse_args()

    try:
        # Create translator instance
        translator = TCLTranslator(args.config)

        # Translate file
        warnings = translator.translate_file(args.input, args.output)

        # Display results
        output_file = args.output or str(
            Path(args.input).with_suffix(".translated" + Path(args.input).suffix)
        )
        print(f"Translation finished! Output file: {output_file}")

        # Display warnings
        if warnings:
            print(f"\n({len(warnings)}) warnings occurred:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print("No warnings")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
