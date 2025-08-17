#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于PLY的Tcl替换脚本
支持根据TOML配置文件对Tcl函数名和选项进行精确替换
使用语法分析而非简单字符串替换
"""

import ply.lex as lex
import ply.yacc as yacc
import toml
import sys
import re
from typing import Dict, List, Optional, Tuple, Any


class TclLexer:
    """Tcl词法分析器"""

    def __init__(self, functions_config: Dict):
        self.functions_config = functions_config
        self.function_names = set(functions_config.keys())

        # 动态生成tokens，包含函数名作为关键字
        self.tokens = (
            "IDENTIFIER",  # 普通标识符
            "FUNCTION",  # 配置中的函数名（关键字）
            "STRING",  # 字符串（带引号）
            "COMMAND_SUB",  # 命令替换 [...]
            "RESERVED",  # 保留字（如if, else等）
            # "LBRACE",  # {
            # "RBRACE",  # }
            "SEMICOLON",  # ;
            "OPTION",  # 选项（以-开头的参数）
            "NEWLINE",  # 换行符
            # "NUMBER",  # 数字
            # "OTHER",  # 其他字符
        )

        self.lexer = None

    def t_COMMAND_SUB(self, t):
        r"\["
        # 处理嵌套的命令替换
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

    def t_STRING(self, t):
        r'"([^"\\]|\\.)*"'
        return t

    def t_OPTION(self, t):
        r"-[a-zA-Z_][a-zA-Z0-9_]+"
        return t

    # def t_NUMBER(self, t):
    #     r"\d+(\.\d+)?"
    #     return t

    def t_IDENTIFIER(self, t):
        # r"[a-zA-Z0-9_.\*/!=$(,)&-]+"
        r'[^"\[\]; \t\r\n]+'
        # 检查是否是配置中的函数名
        if t.value in ["else", "elseif", "if", "while", "for", "proc"]:
            t.type = "RESERVED"  # 保留字
        elif t.value in self.function_names:
            t.type = "FUNCTION"
        return t

    # def t_LBRACE(self, t):
    #     r"\{"
    #     return t
    #
    # def t_RBRACE(self, t):
    #     r"\}"
    #     return t

    def t_SEMICOLON(self, t):
        r";"
        return t

    def t_NEWLINE(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)
        return t

    # def t_OTHER(self, t):
    #     r'[^\s\[\]{};"a-zA-Z_0-9]+'
    #     return t

    # 忽略空白字符（除了换行符）
    t_ignore = " \t\r"

    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
        t.lexer.skip(1)

    def build(self):
        """构建词法分析器"""
        self.lexer = lex.lex(module=self)
        return self.lexer


class TclParser:
    """Tcl语法分析器"""

    def __init__(self, functions_config: Dict):
        self.functions_config = functions_config
        self.tokens = None  # 将由lexer设置

    def p_tcl_script(self, p):
        """tcl_script : statement_list
        | statement_list separator"""
        if len(p) == 2:
            p[0] = self._join_statements(p[1])
        else:
            p[0] = self._join_statements(p[1]) + p[2]

    def p_statement_list_multiple(self, p):
        """statement_list : statement
        | statement_list separator statement"""
        print("processing multiple statements, length:", len(p))
        if len(p) == 2:
            print(f"  Single statement: {p[1]}")
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

        # 应用替换规则
        if func_name in self.functions_config:
            print(f"Processing function call: {func_name} with args: {args}")
            func_info = self.functions_config[func_name]
            new_name = func_info.get("replace_name", func_name)
            param_count = func_info.get("param_count", 0)
            options_config = func_info.get("options", {})

            # 处理参数和选项
            processed_args = []
            current_params = 0

            for arg in args:
                if arg.startswith("-") and arg in options_config:
                    # 替换选项名
                    new_option = options_config[arg].get("replace_name", arg)
                    processed_args.append(new_option)
                else:
                    # 普通参数
                    processed_args.append(arg)
                    if not arg.startswith("-"):
                        current_params += 1

            # 构建替换后的函数调用
            result = new_name
            if processed_args:
                result += " " + " ".join(processed_args)
            p[0] = result
        else:
            # 不在配置中的函数，保持原样
            result = func_name
            if args:
                result += " " + " ".join(args)
            p[0] = result

    def p_expression_standalone_list(self, p):
        """expression_standalone_list : expression_standalone
        | expression_standalone_list expression_standalone"""
        """独立的表达式列表"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_expression_standalone(self, p):
        """expression_standalone : STRING
        | RESERVED
        | IDENTIFIER
        | OPTION"""
        # 独立的表达式（不作为参数）
        p[0] = p[1]

    def p_expression_standalone_command_sub(self, p):
        """expression_standalone : COMMAND_SUB"""
        # 独立的命令替换
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
        # 递归处理命令替换内部的内容
        command_sub_content = p[1]
        print(f"Processing command substitution: {command_sub_content}")
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
        # 处理多个连续分隔符
        p[0] = p[1] + p[2]

    def p_empty(self, p):
        """empty :"""
        pass

    def p_error(self, p):
        if p:
            print(f"Syntax error at token {p.type} ('{p.value}') at line {p.lineno}")
        else:
            print("Syntax error at EOF")

    def _join_statements(self, statements: List[str]) -> str:
        """将语句列表连接成字符串"""
        return " ".join(statements)

    def _process_command_substitution(self, inner_content: str) -> str:
        """递归处理命令替换内部的内容"""
        try:
            # 直接使用当前实例递归处理
            result = (
                self._replacer_instance.parse_and_replace(inner_content)
                if hasattr(self, "_replacer_instance")
                else inner_content
            )
            return result if result else inner_content
        except Exception as e:
            # 如果解析失败，返回原内容
            print(f"Warning: Failed to parse command substitution content: {e}")
            return inner_content

    def build(self, tokens, replacer_instance=None):
        """构建语法分析器"""
        self.tokens = tokens
        if replacer_instance:
            self._replacer_instance = replacer_instance
        self.parser = yacc.yacc(module=self, debug=True, write_tables=False)
        return self.parser


class TclReplacer:
    def __init__(self, config_file: str):
        """初始化替换器

        Args:
            config_file: TOML配置文件路径
        """
        self.config_file = config_file
        self.config = toml.load(config_file)
        self.functions = self.config.get("functions", {})

        # 构建词法和语法分析器
        self.lexer_obj = TclLexer(self.functions)
        self.lexer = self.lexer_obj.build()

        self.parser_obj = TclParser(self.functions)
        self.parser = self.parser_obj.build(self.lexer_obj.tokens, self)

    def process_command_substitution(self, command_sub: str) -> str:
        """处理命令替换，递归解析内部命令"""
        if len(command_sub) <= 2:  # 空的 []
            return command_sub

        inner_content = command_sub[1:-1]  # 去掉 [ 和 ]

        try:
            # 递归解析内部内容
            inner_parser = TclReplacer(self.config_file)
            processed_inner = inner_parser.debug_parse(inner_content)
            # processed_inner = self.parse_and_replace(inner_content)
            return "[" + processed_inner + "]"
        except Exception as e:
            print(
                f"Warning: Failed to process command substitution '{command_sub}': {e}"
            )
            return command_sub

    def tokenize(self, code: str) -> List[Any]:
        """对Tcl代码进行词法分析"""
        self.lexer.input(code)
        tokens = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            tokens.append(tok)
        return tokens

    def parse_and_replace(self, code: str) -> str:
        """解析并替换Tcl代码"""
        try:
            # result = self.parser.parse(code, lexer=self.lexer, debug=1)
            result = self.parser.parse(code, lexer=self.lexer)
            return result if result else code
        except Exception as e:
            print(f"Parse error: {e}")
            return code

    def replace_file(self, input_file: str, output_file: str):
        """替换文件中的内容"""
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 先显示token分析
            print("Token分析:")
            tokens = self.tokenize(content)
            for i, token in enumerate(tokens):
                print(f"  {i+1:2d}. {token.type:12s}: '{token.value}'")

            print("\n语法分析和替换:")
            result = self.parse_and_replace(content)

            print(f"替换后: {result}")
            print("=" * 50)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)

            print(f"替换完成: {input_file} -> {output_file}")

        except Exception as e:
            print(f"处理文件时出错: {e}")

    def debug_tokens(self, code: str):
        """调试用：打印所有token"""
        print(f"调试tokens for: {code}")
        tokens = self.tokenize(code)
        for token in tokens:
            print(f"  {token.type}: '{token.value}' at line {token.lineno}")

    def debug_parse(self, code: str):
        """调试用：解析并打印结果"""
        print(f"原始代码: {code}")
        print("=" * 50)

        # 先显示token分析
        print("Token分析:")
        tokens = self.tokenize(code)
        for i, token in enumerate(tokens):
            print(f"  {i+1:2d}. {token.type:12s}: '{token.value}'")

        print("\n语法分析和替换:")
        result = self.parse_and_replace(code)
        print(f"替换后: {result}")
        print("=" * 50)

        # 测试命令替换递归处理
        if "[" in code and "]" in code:
            print("✓ 检测到命令替换，已启用递归处理")
        return result


def create_sample_config():
    """创建示例配置文件"""
    sample_config = """
# Tcl函数替换配置示例

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
    print("已创建示例配置文件: sample_config.toml")


def create_sample_tcl():
    """创建示例Tcl文件用于测试"""
    sample_tcl = """# 示例Tcl代码
old_proc -help arg1 arg2
test_func -debug param1; legacy_cmd -force

# 测试命令替换
set result [old_proc -verbose [test_func -quiet inner_arg]]
puts $result

# 测试嵌套命令替换
set complex [old_proc -output [legacy_cmd -force] [test_func -debug nested]]

# 普通命令（不会被替换）
other_command -some_option value
"""
    with open("sample_input.tcl", "w", encoding="utf-8") as f:
        f.write(sample_tcl)
    print("已创建示例Tcl文件: sample_input.tcl")


def main():
    if len(sys.argv) < 2:
        print("用法: python tcl_replacer.py <config.toml> <input.tcl> <output.tcl>")
        print("或者: python tcl_replacer.py --create-sample  # 创建示例配置和测试文件")
        print(
            '或者: python tcl_replacer.py --debug <config.toml> "tcl_code"  # 调试模式'
        )
        sys.exit(1)

    if sys.argv[1] == "--create-sample":
        create_sample_config()
        create_sample_tcl()
        print("\n示例用法:")
        print("python tcl_replacer.py sample_config.toml sample_input.tcl output.tcl")
        print(
            'python tcl_replacer.py --debug sample_config.toml "old_proc -help [test_func -debug arg]"'
        )
        return

    if sys.argv[1] == "--debug" and len(sys.argv) >= 4:
        config_file = sys.argv[2]
        test_code = sys.argv[3]
        try:
            replacer = TclReplacer(config_file)
            replacer.debug_parse(test_code)
        except Exception as e:
            print(f"调试错误: {e}")
            import traceback

            traceback.print_exc()
        return

    if len(sys.argv) != 4:
        print("用法: python tcl_replacer.py <config.toml> <input.tcl> <output.tcl>")
        sys.exit(1)

    config_file, input_file, output_file = sys.argv[1:4]

    try:
        replacer = TclReplacer(config_file)
        replacer.replace_file(input_file, output_file)
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
