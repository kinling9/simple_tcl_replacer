#!/usr/bin/env python3
"""
TCL脚本翻译器
支持根据配置文件翻译TCL命令，处理嵌套调用和错误报告
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
    """翻译规则数据类"""

    original_cmd: str
    target_cmd: str
    param_mappings: Dict[str, str] = None  # 参数映射，如 {"-name": "-by_name"}

    def __post_init__(self):
        if self.param_mappings is None:
            self.param_mappings = {}


class TCLTranslator:
    """TCL脚本翻译器主类"""

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
        """加载TOML配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        self._parse_config_data(config)

    def _parse_config_data(self, config: Dict[str, Any]):
        """解析配置数据"""
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
        解析TCL命令行，返回命令和参数列表
        处理嵌套的方括号调用
        """
        commands = []
        i = 0

        while i < len(line):
            # 跳过空白字符
            while i < len(line) and line[i].isspace():
                i += 1

            if i >= len(line):
                break

            # 查找命令的开始
            if line[i] == "[":
                # 处理嵌套命令
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
                    # 未闭合的括号
                    break
            else:
                # 处理普通命令
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
        """解析单个命令行"""
        if not cmd_line.strip():
            return None

        # 使用正则表达式解析命令和参数，考虑引号
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
        """翻译单个命令"""
        # 检查是否为原生TCL命令
        if command in self.native_tcl_commands:
            return command, args

        # 检查是否有翻译规则
        if command not in self.translation_rules:
            return command, args  # 返回原命令，由调用者决定是否报警

        rule = self.translation_rules[command]
        new_command = rule.target_cmd
        new_args = []

        # 翻译参数
        i = 0
        while i < len(args):
            arg = args[i]

            # 检查是否为参数标志（以-开头）
            if arg.startswith("-") and arg in rule.param_mappings:
                new_args.append(rule.param_mappings[arg])
                # 如果有参数值，也要添加
                if i + 1 < len(args):
                    i += 1
                    new_args.append(args[i])
            else:
                new_args.append(arg)

            i += 1

        return new_command, new_args

    def translate_line(self, line: str, line_num: int) -> Tuple[str, List[str]]:
        """
        翻译一行TCL代码
        返回翻译后的行和警告列表
        """
        warnings = []

        # 解析命令
        commands = self.parse_tcl_command(line.strip())

        if not commands:
            return line, warnings

        translated_parts = []
        original_line = line

        for command, args in commands:
            # 翻译命令
            if command in self.native_tcl_commands:
                # 原生命令，不翻译
                translated_cmd = command
                translated_args = args
            elif command in self.translation_rules:
                # 有翻译规则
                translated_cmd, translated_args = self.translate_command(command, args)
            else:
                # 没有翻译规则的非原生命令
                warnings.append(f"line: {line_num}: unsupport command: '{command}'")
                translated_cmd = command
                translated_args = args

            # 重构命令字符串
            if translated_args:
                cmd_str = f"{translated_cmd} {' '.join(translated_args)}"
            else:
                cmd_str = translated_cmd

            translated_parts.append(cmd_str)

        # 重构整行，保持原有的嵌套结构
        if len(commands) == 1:
            # 单个命令
            result = translated_parts[0]
        else:
            # 多个命令，需要保持嵌套结构
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
        """重构包含嵌套命令的行"""
        # 简化处理：对于嵌套命令，直接替换每个识别到的命令
        result = original_line

        for i, (original_cmd, original_args) in enumerate(parsed_commands):
            if i < len(translated_parts):
                # 构建原始命令字符串
                original_cmd_str = (
                    f"{original_cmd} {' '.join(original_args)}"
                    if original_args
                    else original_cmd
                )

                # 在结果中替换
                result = result.replace(
                    f"[{original_cmd_str}]", f"[{translated_parts[i]}]"
                )
                result = result.replace(original_cmd_str, translated_parts[i])

        return result

    def translate_file(self, input_file: str, output_file: str = None) -> List[str]:
        """翻译整个TCL文件"""
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

        # 写入翻译后的文件
        with open(output_file, "w", encoding="utf-8") as f:
            for line in translated_lines:
                f.write(line + "\n")

        return all_warnings


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TCL脚本翻译器")
    parser.add_argument("--config", "-c", required=True, help="配置文件路径")
    parser.add_argument("--input", "-i", required=True, help="输入TCL文件")
    parser.add_argument("--output", "-o", help="输出文件路径（可选）")

    args = parser.parse_args()

    try:
        # 创建翻译器实例
        translator = TCLTranslator(args.config)

        # 翻译文件
        warnings = translator.translate_file(args.input, args.output)

        # 显示结果
        output_file = args.output or str(
            Path(args.input).with_suffix(".translated" + Path(args.input).suffix)
        )
        print(f"translate finished! output_file: {output_file}")

        # 显示警告
        if warnings:
            print(f"\n({len(warnings)}) warnings occured:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print("no warnings")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
