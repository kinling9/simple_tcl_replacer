#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY-based Tcl replacement script
SupportsTcl function names and options with precise replacement based on TOML configuration files
Uses syntax analysis instead of simple string replacement
"""

import ply.lex as lex
import ply.yacc as yacc
import toml
import sys
import re
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TclLexer:
    """Tcl lexer"""

    def __init__(self, functions_config: Dict):
        self.functions_config = functions_config
        self.function_names = set(functions_config.keys())

        # Dynamically generate tokens, including function names as keywords
        self.tokens = (
            "IDENTIFIER",  # Regular identifier
            "FUNCTION",  # Function names from config (keywords)
            "STRING",  # String (quoted)
            "COMMAND_SUB",  # Command substitution [...]
            "RESERVED",  # Reserved words (e.g., if, else, etc.)
            "SEMICOLON",  # ;
            "OPTION",  # Option (parameter starting with -)
            "NEWLINE",  # Newline character
        )

        self.lexer = None

    def t_COMMAND_SUB(self, t):
        r"\["
        # Handle nested command substitutions
        bracket_count = 1
        start_pos = t.lexpos
        i = start_pos + 1

        while i < len(t.lexer.lexdata) and bracket_count > 0:
            char = t.lexer.lexdata[i]
            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
            i += 1

        if bracket_count == 0:
            t.value = t.lexer.lexdata[start_pos:i]
            t.lexer.lexpos = i
        return t

    def t_STRING_OR_QUOTE(self, t):
        r'"'
        # Look ahead to determine if this is a complete string or just a quote
        start_pos = t.lexpos
        i = start_pos + 1

        # Scan forward to see what's inside the quotes
        while i < len(t.lexer.lexdata):
            char = t.lexer.lexdata[i]

            if char == '"':
                # Found closing quote - this is a complete string
                t.type = "STRING"
                t.value = t.lexer.lexdata[start_pos : i + 1]
                t.lexer.lexpos = i + 1
                self.lexer.lasttoken = t
                return t
            elif char == "\\":
                i += 2  # Skip escaped character
            elif char == "[":
                # Found '[' inside quotes - check if it should trigger command substitution
                # Command substitution triggers if '[' appears:
                # 1. Immediately after opening quote, OR
                # 2. After whitespace characters (space, tab, newline, etc.)
                if i == start_pos + 1:  # Immediately after opening quote
                    t.type = "IDENTIFIER"
                    t.value = '"'
                    self.lexer.lasttoken = t
                    return t
                elif (
                    i > start_pos + 1 and t.lexer.lexdata[i - 1] in " \t\r\n\f\v"
                ):  # After whitespace
                    t.type = "IDENTIFIER"
                    t.value = '"'
                    self.lexer.lasttoken = t
                    return t
                else:
                    i += 1
            elif char == "\n" or char == "\r":
                # Unterminated string - treat as quote
                t.type = "IDENTIFIER"
                t.value = '"'
                self.lexer.lasttoken = t
                return t
            else:
                i += 1

        # Reached end of input without closing quote - treat as quote
        t.type = "IDENTIFIER"
        t.value = '"'
        self.lexer.lasttoken = t
        return t

    def t_OPTION(self, t):
        r"-[a-zA-Z_][a-zA-Z0-9_]+"
        return t

    def t_IDENTIFIER(self, t):
        r'[^"\[\]; \t\r\n]+(?:\[[^"\[\]; \t\r\n]*\])*'

        # Get the last token (if any)
        if hasattr(self.lexer, "lasttoken"):
            # If last token was "set", this must be an identifier
            if (
                self.lexer.lasttoken != None
                and self.lexer.lasttoken.type == "IDENTIFIER"
                and self.lexer.lasttoken.value == "set"
            ):
                return t

        # Check if it's a function name from config
        if t.value in ["else", "elseif", "if", "while", "for", "proc"]:
            t.type = "RESERVED"  # Reserved word
        elif t.value in self.function_names:
            t.type = "FUNCTION"

        # backup the last token
        self.lexer.lasttoken = t
        return t

    def t_SEMICOLON(self, t):
        r";"
        return t

    def t_NEWLINE(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)
        return t

    # Ignore whitespace characters (except newlines)
    t_ignore = " \t\r"

    def t_error(self, t):
        logging.error(f"Illegal character '{t.value[0]}' at line {t.lineno}")
        t.lexer.skip(1)

    def build(self):
        """Build the lexer"""
        self.lexer = lex.lex(module=self)
        self.lexer.lasttoken = None
        return self.lexer


class TclParser:
    """Tcl parser"""

    def __init__(self, functions_config: Dict):
        self.functions_config = functions_config
        self.tokens = None  # Will be set by lexer

    def p_tcl_script(self, p):
        """tcl_script : statement_list
        | statement_list separator
        | separator statement_list
        | separator statement_list separator
        | separator"""
        if len(p) == 2:
            if isinstance(p[1], list):  # statement_list
                p[0] = self._join_statements(p[1])
            else:  # separator only
                p[0] = p[1]
        elif len(p) == 3:
            if isinstance(p[1], list):  # statement_list separator
                p[0] = self._join_statements(p[1]) + p[2]
            else:  # separator statement_list
                p[0] = p[1] + self._join_statements(p[2])
        else:  # separator statement_list separator
            p[0] = p[1] + self._join_statements(p[2]) + p[3]

    def p_statement_list_multiple(self, p):
        """statement_list : statement
        | statement_list separator statement"""
        logging.debug("Processing multiple statements, length: %d", len(p))
        if len(p) == 2:
            logging.debug("  Single statement: %s", p[1])
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]] + [p[3]]

    def p_statement(self, p):
        """statement : function_call
        | expression_standalone_list"""
        if isinstance(p[1], list):
            p[0] = self._join_statements(p[1])
        else:
            p[0] = p[1]

    def p_function_call(self, p):
        """function_call : FUNCTION argument_list"""
        func_name = p[1]
        args = p[2]

        # Apply replacement rules
        if func_name in self.functions_config:
            logging.info(f"Processing function call: {func_name} with args: {args}")
            func_info = self.functions_config[func_name]
            new_name = func_info.get("replace_name", func_name)
            param_count = func_info.get("param_count", 0)
            options_config = func_info.get("options", {})
            params_config = func_info.get("params", {})

            # Process arguments and options
            processed_args = []
            current_params = 0

            for arg in args:
                if arg.startswith("-") and arg in options_config:
                    # Replace option name
                    new_option = options_config[arg].get("replace_name", arg)
                    processed_args.append(new_option)
                else:
                    # Regular parameter
                    params_matched = False
                    for param_pattern, param_info in params_config.items():
                        replace_pattern = param_info.get("replace_pattern", None)
                        if replace_pattern and re.match(param_pattern, arg):
                            new_param = re.sub(param_pattern, replace_pattern, arg)
                            processed_args.append(new_param)
                            params_matched = True
                            if not arg.startswith("-"):
                                current_params += 1
                            break
                    if not params_matched:
                        processed_args.append(arg)
                    if not arg.startswith("-"):
                        current_params += 1

            # Build replaced function call
            result = new_name
            if processed_args:
                result += " " + " ".join(processed_args)
            p[0] = result
        else:
            # Function not in config, keep as is
            result = func_name
            if args:
                result += " " + " ".join(args)
            p[0] = result

    def p_expression_standalone_list(self, p):
        """expression_standalone_list : expression_standalone
        | expression_standalone_list expression_standalone"""
        """List of standalone expressions"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_expression_standalone(self, p):
        """expression_standalone : STRING
        | RESERVED
        | IDENTIFIER
        | OPTION"""
        # Standalone expression (not as parameter)
        p[0] = p[1]

    def p_expression_standalone_command_sub(self, p):
        """expression_standalone : COMMAND_SUB"""
        # Standalone command substitution
        command_sub_content = p[1]
        if hasattr(self, "_replacer_instance"):
            processed = self._replacer_instance.process_command_substitution(
                command_sub_content
            )
            p[0] = processed
        else:
            p[0] = command_sub_content

    def p_argument_list_multiple(self, p):
        """argument_list : argument_list argument"""
        p[0] = p[1] + [p[2]]

    def p_argument_list_single(self, p):
        """argument_list : argument"""
        p[0] = [p[1]]

    def p_argument_list_empty(self, p):
        """argument_list : empty"""
        p[0] = []

    def p_argument(self, p):
        """argument : IDENTIFIER
        | STRING
        | OPTION"""
        p[0] = p[1]

    def p_argument_command_sub(self, p):
        """argument : COMMAND_SUB"""
        # Recursively process content inside command substitution
        command_sub_content = p[1]
        logging.info(f"Processing command substitution: {command_sub_content}")
        if hasattr(self, "_replacer_instance"):
            processed = self._replacer_instance.process_command_substitution(
                command_sub_content
            )
            p[0] = processed
        else:
            p[0] = command_sub_content

    def p_separator(self, p):
        """separator : SEMICOLON
        | NEWLINE"""
        p[0] = p[1]

    def p_separator_multiple(self, p):
        """separator : separator SEMICOLON
        | separator NEWLINE"""
        # Handle multiple consecutive separators
        p[0] = p[1] + p[2]

    def p_empty(self, p):
        """empty :"""
        pass

    def p_error(self, p):
        if p:
            logging.error(
                f"Syntax error at token {p.type} ('{p.value}') at line {p.lineno}"
            )
        else:
            logging.error("Syntax error at EOF")

    def _join_statements(self, statements: List[str]) -> str:
        """Join list of statements into a string"""
        return " ".join(statements)

    def _process_command_substitution(self, inner_content: str) -> str:
        """Recursively process content inside command substitution"""
        try:
            # Use current instance for recursive processing
            result = (
                self._replacer_instance.parse_and_replace(inner_content)
                if hasattr(self, "_replacer_instance")
                else inner_content
            )
            return result if result else inner_content
        except Exception as e:
            # If parsing fails, return original content
            logging.warning(f"Failed to parse command substitution content: {e}")
            return inner_content

    def build(self, tokens, replacer_instance=None):
        """Build the parser"""
        self.tokens = tokens
        if replacer_instance:
            self._replacer_instance = replacer_instance
        self.parser = yacc.yacc(module=self, debug=True, write_tables=False)
        return self.parser


class TclReplacer:
    def __init__(self, config_file: str):
        """Initialize the replacer

        Args:
            config_file: Path to TOML configuration file
        """
        self.config_file = config_file
        self.config = toml.load(config_file)
        self.functions = self.config.get("functions", {})

        # Build lexer and parser
        self.lexer_obj = TclLexer(self.functions)
        self.lexer = self.lexer_obj.build()

        self.parser_obj = TclParser(self.functions)
        self.parser = self.parser_obj.build(self.lexer_obj.tokens, self)

    def process_command_substitution(self, command_sub: str) -> str:
        """Process command substitution, recursively parse inner commands"""
        if len(command_sub) <= 2:  # Empty []
            return command_sub

        inner_content = command_sub[1:-1]  # Remove [ and ]

        try:
            # Recursively parse inner content
            inner_parser = TclReplacer(self.config_file)
            processed_inner = inner_parser.debug_parse(inner_content)
            return "[" + processed_inner + "]"
        except Exception as e:
            logging.warning(
                f"Failed to process command substitution '{command_sub}': {e}"
            )
            return command_sub

    def tokenize(self, code: str) -> List[Any]:
        """Perform lexical analysis on Tcl code"""
        self.lexer.input(code)
        tokens = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            tokens.append(tok)
        return tokens

    def parse_and_replace(self, code: str) -> str:
        """Parse and replace Tcl code"""
        try:
            result = self.parser.parse(code, lexer=self.lexer)
            return result if result else code
        except Exception as e:
            logging.error(f"Parse error: {e}")
            return code

    def replace_file(self, input_file: str, output_file: str):
        """Replace content in file"""
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()

            logging.info("\nSyntax analysis and replacement:")
            result = self.parse_and_replace(content)

            logging.info(f"After replacement: {result}")
            logging.info("=" * 50)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)

            logging.info(f"Replacement completed: {input_file} -> {output_file}")

        except Exception as e:
            logging.error(f"Error processing file: {e}")

    def debug_tokens(self, code: str):
        """For debugging: print all tokens"""
        logging.debug(f"Debug tokens for: {code}")
        tokens = self.tokenize(code)
        for token in tokens:
            logging.debug(f"  {token.type}: '{token.value}' at line {token.lineno}")

    def debug_parse(self, code: str):
        """For debugging: parse and print result"""
        logging.info(f"Original code: {code}")
        logging.info("=" * 50)

        logging.info("\nSyntax analysis and replacement:")
        result = self.parse_and_replace(code)
        logging.info(f"After replacement: {result}")
        logging.info("=" * 50)

        # Test recursive processing of command substitution
        if "[" in code and "]" in code:
            logging.info(
                "✓ Detected command substitution, recursive processing enabled"
            )
        return result


def create_sample_config():
    """Create sample configuration file"""
    sample_config = """
# Tcl function replacement configuration example

[functions.old_proc]
replace_name = "new_proc"
param_count = 2

[functions.old_proc.options]
"-help" = { replace_name = "-h" }
"-verbose" = { replace_name = "-v" }
"-output" = { replace_name = "-o" }

[functions.test_func]
replace_name = "better_func"
param_count = 1

[functions.test_func.options]
"-debug" = { replace_name = "-d" }
"-quiet" = { replace_name = "-q" }

[functions.legacy_cmd]
replace_name = "modern_cmd"
param_count = 0

[functions.legacy_cmd.options]
"-force" = { replace_name = "-f" }
"""
    with open("sample_config.toml", "w", encoding="utf-8") as f:
        f.write(sample_config)
    logging.info("Created sample configuration file: sample_config.toml")


def create_sample_tcl():
    """Create sample Tcl file for testing"""
    sample_tcl = """# Sample Tcl code
old_proc -help arg1 arg2
test_func -debug param1; legacy_cmd -force

# Test command substitution
set result [old_proc -verbose [test_func -quiet inner_arg]]
puts $result

# Test nested command substitution
set complex [old_proc -output [legacy_cmd -force] [test_func -debug nested]]

# Regular command (will not be replaced)
other_command -some_option value
"""
    with open("sample_input.tcl", "w", encoding="utf-8") as f:
        f.write(sample_tcl)
    logging.info("Created sample Tcl file: sample_input.tcl")


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Tcl function and option replacement tool based on TOML configuration"
    )

    # Add mutually exclusive group for commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample configuration and test files",
    )
    group.add_argument(
        "--debug",
        nargs=2,
        metavar=("CONFIG", "CODE"),
        help="Debug mode: parse and replace given Tcl code using specified config",
    )
    group.add_argument(
        "--process",  # Added a flag to make this optional
        nargs=3,
        metavar=("CONFIG", "INPUT", "OUTPUT"),
        help="Path to config file, input Tcl file, and output file",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.create_sample:
        create_sample_config()
        create_sample_tcl()
        logging.info("\nExample usage:")
        logging.info(
            "python tcl_replacer.py --proces sample_config.toml sample_input.tcl output.tcl"
        )
        logging.info(
            'python tcl_replacer.py --debug sample_config.toml "old_proc -help [test_func -debug arg]"'
        )
        return

    if args.debug:
        config_file, test_code = args.debug
        try:
            replacer_token = TclReplacer(config_file)
            replacer_token.debug_tokens(test_code)
            replacer = TclReplacer(config_file)
            replacer.debug_parse(test_code)
        except Exception as e:
            logging.error(f"Debug error: {e}")
            import traceback

            traceback.print_exc()
        return

    # Process file replacement
    config_file, input_file, output_file = args.process
    try:
        replacer_token = TclReplacer(config_file)

        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
            replacer_token.debug_tokens(content)
        replacer = TclReplacer(config_file)
        replacer.replace_file(input_file, output_file)
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
