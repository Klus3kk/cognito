"""
General purpose code correction system for Cognito.
"""

import ast
import re
from difflib import unified_diff
from typing import Dict, List, Tuple, Any, Optional, Union

class CodeCorrector:
    """Provides automatic code correction functionality for various languages"""
    
    def __init__(self, language: str = "python"):
        """
        Initialize the code corrector.
        
        Args:
            language: The programming language of the code to correct
        """
        self.language = language.lower()
        self.corrections = []
        
    def correct_code(self, code: str, issues: List[Dict[str, Any]]) -> str:
        """
        Generate corrected code based on identified issues.
        
        Args:
            code: The original source code
            issues: List of issues with line numbers and suggestions
            
        Returns:
            Corrected code
        """
        if self.language == "python":
            return self._correct_python_code(code, issues)
        elif self.language == "c":
            return self._correct_c_code(code, issues)
        else:
            return "# Automatic correction not supported for this language yet\n\n" + code
    
    def _correct_python_code(self, code: str, issues: List[Dict[str, Any]]) -> str:
        """Correct Python code based on identified issues"""
        # Analyze code structure to improve corrections
        try:
            tree = ast.parse(code)
            # Set parent references to improve context
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    setattr(child, 'parent', node)
        except:
            # If parsing fails, we'll do our best without the AST
            tree = None
        
        # Convert code to lines for easier manipulation
        lines = code.split('\n')
        corrected_lines = lines.copy()
        
        # Track line offsets as we add/remove lines
        line_offset = 0
        
        # Group issues by type for better correction patterns
        security_issues = []
        performance_issues = []
        readability_issues = []
        other_issues = []
        
        for issue in issues:
            issue_type = issue.get("type", "").lower()
            
            if "security" in issue_type or any(kw in issue.get("message", "").lower() for kw in 
                                              ["security", "eval", "open", "injection", "vulnerability"]):
                security_issues.append(issue)
            elif "performance" in issue_type or any(kw in issue.get("message", "").lower() for kw in 
                                                  ["performance", "complexity", "nested", "recursion"]):
                performance_issues.append(issue)
            elif "readability" in issue_type or any(kw in issue.get("message", "").lower() for kw in 
                                                  ["readability", "style", "indent", "docstring"]):
                readability_issues.append(issue)
            else:
                other_issues.append(issue)
        
        # Process security issues first (highest priority)
        corrected_lines, line_offset = self._process_security_issues(corrected_lines, security_issues, line_offset)
        
        # Process performance issues next
        corrected_lines, line_offset = self._process_performance_issues(corrected_lines, performance_issues, line_offset, tree)
        
        # Process readability issues
        corrected_lines, line_offset = self._process_readability_issues(corrected_lines, readability_issues, line_offset)
        
        # Process other issues
        corrected_lines, line_offset = self._process_other_issues(corrected_lines, other_issues, line_offset)
        
        # Join the corrected lines back together
        return "\n".join(corrected_lines)
    
    def _process_security_issues(self, lines: List[str], issues: List[Dict[str, Any]], line_offset: int) -> Tuple[List[str], int]:
        """Process security-related issues in the code"""
        for issue in sorted(issues, key=lambda x: x.get("line", 0)):
            line_num = issue.get("line", 0)
            if line_num <= 0:
                continue  # Skip issues without line numbers
                
            # Adjust line number based on previous corrections
            adjusted_line = line_num + line_offset
            if adjusted_line <= 0 or adjusted_line > len(lines):
                continue
                
            # Get the current line
            line = lines[adjusted_line - 1]
            
            # Check what kind of security issue it is
            message = issue.get("message", "").lower()
            func_name = issue.get("func_name", "")
            
            if "eval" in message or "eval" == func_name:
                # Handle eval security issue
                correction = self._fix_eval_issue(line, issue)
                if correction:
                    if isinstance(correction, list):
                        # Replace line with multiple lines
                        lines[adjusted_line - 1] = correction[0]
                        for i, new_line in enumerate(correction[1:], 1):
                            lines.insert(adjusted_line - 1 + i, new_line)
                        line_offset += len(correction) - 1
                    else:
                        # Simple replacement
                        lines[adjusted_line - 1] = correction
            
            elif "open" in message or "open" == func_name:
                # Handle open security issue
                # First, check if we can find the subsequent usage of the file handle
                correction = self._fix_open_issue(lines, adjusted_line - 1, issue)
                if correction:
                    # If there is a multi-line replacement
                    if isinstance(correction, list):
                        # Comment out original line
                        indent = len(line) - len(line.lstrip())
                        indent_str = " " * indent
                        lines[adjusted_line - 1] = f"{indent_str}# FIXED: {line.lstrip()}"
                        
                        # Insert correction lines
                        for i, new_line in enumerate(correction):
                            lines.insert(adjusted_line + i, new_line)
                        
                        # Update line offset
                        line_offset += len(correction)
                    else:
                        # Simple replacement
                        lines[adjusted_line - 1] = correction
            
            elif "system" in message or "system" in func_name:
                # Handle os.system security issue
                correction = self._fix_system_issue(line, issue)
                if correction:
                    if isinstance(correction, list):
                        lines[adjusted_line - 1] = correction[0]
                        for i, new_line in enumerate(correction[1:], 1):
                            lines.insert(adjusted_line - 1 + i, new_line)
                        line_offset += len(correction) - 1
                    else:
                        lines[adjusted_line - 1] = correction
        
        return lines, line_offset
    
    def _process_performance_issues(self, lines: List[str], issues: List[Dict[str, Any]], line_offset: int, tree=None) -> Tuple[List[str], int]:
        """Process performance-related issues in the code"""
        for issue in sorted(issues, key=lambda x: x.get("line", 0)):
            line_num = issue.get("line", 0)
            if line_num <= 0:
                continue  # Skip issues without line numbers
                
            # Adjust line number based on previous corrections
            adjusted_line = line_num + line_offset
            if adjusted_line <= 0 or adjusted_line > len(lines):
                continue
                
            # Get the current line
            line = lines[adjusted_line - 1]
            
            # Check what kind of performance issue it is
            message = issue.get("message", "").lower()
            
            if "nested loop" in message:
                # Handle nested loop issue
                correction = self._fix_nested_loop_issue(line, issue)
                if correction:
                    if isinstance(correction, list):
                        # Insert the multi-line correction
                        for i, new_line in enumerate(correction):
                            if i == 0:
                                lines[adjusted_line - 1] = new_line
                            else:
                                lines.insert(adjusted_line - 1 + i, new_line)
                        line_offset += len(correction) - 1
                    else:
                        lines[adjusted_line - 1] = correction
            
            elif "recursion" in message or "recursive" in message:
                # Handle recursion issue - look for function definition
                if "def " in line and "()" in line or "(" in line and ")" in line:
                    correction = self._fix_recursive_function_issue(lines, adjusted_line - 1, issue, tree)
                    if correction:
                        if isinstance(correction, list):
                            # Comment out the original line
                            indent = len(line) - len(line.lstrip())
                            indent_str = " " * indent
                            lines[adjusted_line - 1] = f"{indent_str}# SUGGESTION: Improve recursive function below"
                            
                            # Insert the suggested implementation after the original function
                            # Find the end of the function first
                            func_end = adjusted_line
                            while func_end < len(lines):
                                if not lines[func_end].strip() or lines[func_end].startswith(' ' * (indent + 1)):
                                    func_end += 1
                                else:
                                    break
                            
                            # Add blank line
                            lines.insert(func_end, "")
                            line_offset += 1
                            
                            # Add improved implementation
                            for i, new_line in enumerate(correction):
                                lines.insert(func_end + i + 1, new_line)
                            line_offset += len(correction)
                        else:
                            # Add a comment to the function definition
                            lines[adjusted_line - 1] = f"{line}  # {correction}"
                else:
                    # The line isn't a function definition, just add a comment
                    lines[adjusted_line - 1] = f"{line}  # Consider improving recursive function performance"
            
            elif "string concat" in message:
                # Handle string concatenation issue
                correction = self._fix_string_concatenation_issue(line, issue)
                if correction:
                    if isinstance(correction, list):
                        lines[adjusted_line - 1] = correction[0]
                        for i, new_line in enumerate(correction[1:], 1):
                            lines.insert(adjusted_line - 1 + i, new_line)
                        line_offset += len(correction) - 1
                    else:
                        lines[adjusted_line - 1] = correction
        
        return lines, line_offset
    
    def _process_readability_issues(self, lines: List[str], issues: List[Dict[str, Any]], line_offset: int) -> Tuple[List[str], int]:
        """Process readability-related issues in the code"""
        # Process readability issues in a similar way
        for issue in sorted(issues, key=lambda x: x.get("line", 0)):
            line_num = issue.get("line", 0)
            if line_num <= 0:
                continue
                
            adjusted_line = line_num + line_offset
            if adjusted_line <= 0 or adjusted_line > len(lines):
                continue
                
            line = lines[adjusted_line - 1]
            message = issue.get("message", "").lower()
            
            # Handle indentation issues
            if "indent" in message:
                indent = len(line) - len(line.lstrip())
                # Standardize to 4-space indentation
                correct_indent = (indent // 4) * 4
                if indent != correct_indent:
                    lines[adjusted_line - 1] = " " * correct_indent + line.lstrip()
            
            # Handle docstring issues
            elif "docstring" in message and "def " in line:
                # Add docstring to function
                match = re.search(r"def\s+(\w+)\s*\((.*?)\)", line)
                if match:
                    func_name = match.group(1)
                    params = match.group(2)
                    
                    # Check the next line
                    next_line_idx = adjusted_line
                    if next_line_idx < len(lines):
                        next_line = lines[next_line_idx]
                        if not '"""' in next_line and not "'''" in next_line:
                            # Add docstring after function definition
                            indent = len(line) - len(line.lstrip())
                            docstring = [
                                f"{' ' * (indent + 4)}\"\"\"",
                                f"{' ' * (indent + 4)}{func_name} function.",
                                f"{' ' * (indent + 4)}",
                            ]
                            
                            # Add param descriptions if there are parameters
                            if params.strip():
                                param_list = [p.strip() for p in params.split(",")]
                                if param_list:
                                    docstring.append(f"{' ' * (indent + 4)}Args:")
                                    for param in param_list:
                                        param_name = param.split(":")[0].split("=")[0].strip()
                                        if param_name:
                                            docstring.append(f"{' ' * (indent + 8)}{param_name}: Description of {param_name}")
                            
                            docstring.append(f"{' ' * (indent + 4)}Returns:")
                            docstring.append(f"{' ' * (indent + 8)}Description of return value")
                            docstring.append(f"{' ' * (indent + 4)}\"\"\"")
                            
                            # Insert docstring
                            for i, doc_line in enumerate(docstring):
                                lines.insert(adjusted_line + i, doc_line)
                            line_offset += len(docstring)
            
            # Handle naming convention issues
            elif "naming" in message:
                if "function" in message and "def " in line:
                    # Fix function naming (to snake_case)
                    match = re.search(r"def\s+(\w+)", line)
                    if match:
                        func_name = match.group(1)
                        if not re.match(r"^[a-z][a-z0-9_]*$", func_name):
                            # Convert to snake_case
                            snake_case = re.sub(r"([A-Z])", r"_\1", func_name).lower()
                            if snake_case.startswith("_"):
                                snake_case = snake_case[1:]
                            lines[adjusted_line - 1] = line.replace(f"def {func_name}", f"def {snake_case}")
                elif "class" in message and "class " in line:
                    # Fix class naming (to PascalCase)
                    match = re.search(r"class\s+(\w+)", line)
                    if match:
                        class_name = match.group(1)
                        if not re.match(r"^[A-Z][a-zA-Z0-9]*$", class_name):
                            # Convert to PascalCase
                            pascal_case = "".join(word.capitalize() for word in re.split(r"[_\-]", class_name))
                            lines[adjusted_line - 1] = line.replace(f"class {class_name}", f"class {pascal_case}")
        
        return lines, line_offset
    
    def _process_other_issues(self, lines: List[str], issues: List[Dict[str, Any]], line_offset: int) -> Tuple[List[str], int]:
        """Process other miscellaneous issues"""
        # Handle any other issues
        return lines, line_offset
    
    def _fix_eval_issue(self, line: str, issue: Dict[str, Any]) -> Optional[Union[str, List[str]]]:
        """Fix eval security issue"""
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent
        
        # Try to extract the expression from eval()
        eval_match = re.search(r'eval\s*\(\s*(.+?)\s*\)', line)
        if eval_match:
            expr = eval_match.group(1)
            
            # If it looks like a simple arithmetic expression, suggest ast.literal_eval
            if expr.startswith('"') or expr.startswith("'"):
                return [
                    f"{indent_str}# WARNING: Unsafe eval detected - removing:",
                    f"{indent_str}# {line.strip()}",
                    f"{indent_str}import ast  # Add import at the top of your file",
                    f"{indent_str}processed = ast.literal_eval({expr})  # Safe alternative"
                ]
            
            # If it looks more complex, suggest safer alternatives
            return [
                f"{indent_str}# WARNING: Unsafe eval detected - removing:",
                f"{indent_str}# {line.strip()}",
                f"{indent_str}# Consider using a safer alternative:",
                f"{indent_str}# 1. For math expressions: import ast; ast.literal_eval(expr)",
                f"{indent_str}# 2. For simple operations: directly compute the result",
                f"{indent_str}# 3. For configs: use json.loads() or yaml.safe_load()"
            ]
        
        return None
    
    def _fix_open_issue(self, lines: List[str], line_idx: int, issue: Dict[str, Any]) -> Optional[List[str]]:
        """Fix file handling security issue"""
        line = lines[line_idx]
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent
        
        # Try to extract variable assignment and parameters
        open_match = re.search(r'(\w+)\s*=\s*open\s*\(\s*(.+?)\s*\)', line)
        if open_match:
            var_name = open_match.group(1)
            params = open_match.group(2)
            
            # Look ahead for possible usage of the file handle
            data_var = None
            for i in range(line_idx + 1, min(line_idx + 5, len(lines))):
                next_line = lines[i]
                if not next_line.strip():
                    continue
                
                # Check if it's using the file handle's readlines() method
                readlines_match = re.search(rf'(\w+)\s*=\s*{var_name}\.readlines\(\)', next_line)
                if readlines_match:
                    data_var = readlines_match.group(1)
                    break
                
                # Check if it's using the file handle's read() method
                read_match = re.search(rf'(\w+)\s*=\s*{var_name}\.read\(\)', next_line)
                if read_match:
                    data_var = read_match.group(1)
                    break
            
            if data_var:
                # We found how the file handle is used
                return [
                    f"{indent_str}# FIXED: Replaced unsafe file handling with context manager",
                    f"{indent_str}# Original: {line.strip()}",
                    f"{indent_str}try:",
                    f"{indent_str}    with open({params}) as file_handler:",
                    f"{indent_str}        {data_var} = file_handler.readlines()  # Direct assignment to data variable",
                    f"{indent_str}except FileNotFoundError:",
                    f"{indent_str}    print(f\"File not found: {{{params.split(',')[0]}}}\")  # Handle error appropriately",
                    f"{indent_str}    {data_var} = []  # Provide a default value",
                    f"{indent_str}except IOError as e:",
                    f"{indent_str}    print(f\"I/O error: {{e}}\")  # Handle error appropriately",
                    f"{indent_str}    {data_var} = []  # Provide a default value"
                ]
            else:
                # Generic correction when we can't determine how the file handle is used
                return [
                    f"{indent_str}# FIXED: Replaced unsafe file handling with context manager",
                    f"{indent_str}# Original: {line.strip()}",
                    f"{indent_str}try:",
                    f"{indent_str}    with open({params}) as {var_name}:",
                    f"{indent_str}        # Your file operations here",
                    f"{indent_str}        data = {var_name}.readlines()  # Example operation",
                    f"{indent_str}except FileNotFoundError:",
                    f"{indent_str}    print(f\"File not found: {{{params.split(',')[0]}}}\")  # Handle error appropriately",
                    f"{indent_str}    data = []  # Provide a default value",
                    f"{indent_str}except IOError as e:",
                    f"{indent_str}    print(f\"I/O error: {{e}}\")  # Handle error appropriately",
                    f"{indent_str}    data = []  # Provide a default value"
                ]
        
        return None
    
    def _fix_system_issue(self, line: str, issue: Dict[str, Any]) -> Optional[Union[str, List[str]]]:
        """Fix os.system security issue"""
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent
        
        # Try to extract the command
        system_match = re.search(r'(?:os\.)?system\s*\(\s*(.+?)\s*\)', line)
        if system_match:
            cmd = system_match.group(1)
            
            return [
                f"{indent_str}# WARNING: os.system is a security risk - use subprocess instead:",
                f"{indent_str}# {line.strip()}",
                f"{indent_str}import subprocess  # Add import at the top of your file",
                f"{indent_str}# Using subprocess.run with shell=False prevents command injection",
                f"{indent_str}result = subprocess.run({cmd}.split(), shell=False, capture_output=True, text=True, check=True)"
            ]
        
        return None
    
    def _fix_nested_loop_issue(self, line: str, issue: Dict[str, Any]) -> Optional[Union[str, List[str]]]:
        """Fix nested loop performance issue"""
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent
        
        if "for " in line:
            # Check if it's a numpy-compatible loop (e.g., working with matrices)
            return [
                f"{indent_str}# PERFORMANCE IMPROVEMENT SUGGESTION:",
                f"{indent_str}# The current nested loop structure has high time complexity.",
                f"{indent_str}# Consider using numpy for matrix operations or list comprehensions:",
                f"{indent_str}# import numpy as np",
                f"{indent_str}# result = np.array(matrix).reshape(-1)  # Flatten to 1D",
                f"{indent_str}# Or with list comprehensions:",
                f"{indent_str}# result = [val for row in matrix for val in row]  # Flattens 2D array",
                f"{indent_str}{line.lstrip()}"
            ]
        
        return None
    
    def _fix_recursive_function_issue(self, lines: List[str], line_idx: int, issue: Dict[str, Any], tree=None) -> Optional[Union[str, List[str]]]:
        """Fix recursive function performance issue"""
        line = lines[line_idx]
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent
        
        # Extract function name
        func_match = re.search(r'def\s+(\w+)', line)
        if not func_match:
            return None
        
        func_name = func_match.group(1)
        
        # Check if it seems like a Fibonacci implementation
        is_fibonacci = False
        fib_keywords = ["fibonacci", "fib"]
        
        if any(kw in func_name.lower() for kw in fib_keywords):
            is_fibonacci = True
        else:
            # Check function body for typical Fibonacci pattern
            for i in range(line_idx + 1, min(line_idx + 20, len(lines))):
                check_line = lines[i].lower()
                if indent_str not in check_line:  # End of function
                    break
                if f"{func_name}(n-1)" in check_line and f"{func_name}(n-2)" in check_line:
                    is_fibonacci = True
                    break
        
        if is_fibonacci:
            # Provide Fibonacci-specific optimizations
            return [
                f"{indent_str}def {func_name}_optimized(n):",
                f"{indent_str}    \"\"\"Optimized version of {func_name} using dynamic programming.\"\"\"",
                f"{indent_str}    # Using iterative approach instead of recursion",
                f"{indent_str}    if n <= 0:",
                f"{indent_str}        return []",
                f"{indent_str}    elif n == 1:",
                f"{indent_str}        return [0]",
                f"{indent_str}    elif n == 2:",
                f"{indent_str}        return [0, 1]",
                f"{indent_str}    ",
                f"{indent_str}    fib = [0, 1]",
                f"{indent_str}    for i in range(2, n):",
                f"{indent_str}        fib.append(fib[i-1] + fib[i-2])",
                f"{indent_str}    ",
                f"{indent_str}    return fib",
                f"{indent_str}",
                f"{indent_str}# Alternatively, use memoization to optimize the recursive approach:",
                f"{indent_str}from functools import lru_cache",
                f"{indent_str}",
                f"{indent_str}@lru_cache(maxsize=None)",
                f"{indent_str}def {func_name}_memoized(n):",
                f"{indent_str}    \"\"\"Memoized version of {func_name} using functools.lru_cache.\"\"\"",
                f"{indent_str}    if n <= 0:",
                f"{indent_str}        return 0",
                f"{indent_str}    elif n == 1:",
                f"{indent_str}        return 1",
                f"{indent_str}    else:",
                f"{indent_str}        return {func_name}_memoized(n-1) + {func_name}_memoized(n-2)"
            ]
        else:
            # Generic recursive function optimization
            return [
                f"{indent_str}# OPTIMIZATION SUGGESTION: Add memoization to recursive function",
                f"{indent_str}from functools import lru_cache",
                f"{indent_str}",
                f"{indent_str}@lru_cache(maxsize=None)",
                f"{indent_str}def {func_name}_memoized{line[line.index('('):]}",
                f"{indent_str}    \"\"\"Memoized version of {func_name} using functools.lru_cache.\"\"\"",
                f"{indent_str}    # Same function body as original, but results will be cached"
            ]
    
    def _fix_string_concatenation_issue(self, line: str, issue: Dict[str, Any]) -> Optional[Union[str, List[str]]]:
        """Fix string concatenation performance issue"""
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent
        
        # Check for inefficient concatenation in a loop
        if "for " in line:
            # This might be the loop header
            return f"{line}  # Consider using ''.join() for better performance"
        
        # Check for a += b pattern
        concat_match = re.search(r'(\w+)\s*=\s*\1\s*\+\s*(.+)', line)
        if concat_match:
            var_name = concat_match.group(1)
            added_part = concat_match.group(2)
            
            return [
                f"{indent_str}# FIXED: More efficient string concatenation",
                f"{indent_str}# Original: {line.strip()}",
                f"{indent_str}{var_name} += {added_part}  # Use += for in-place concatenation"
            ]
        
        return None
    
    def _correct_c_code(self, code: str, issues: List[Dict[str, Any]]) -> str:
        """Correct C code based on identified issues"""
        # Similar structure to _correct_python_code but with C-specific fixes
        # Currently, returns the original code
        return "// Automatic C code correction not implemented yet\n\n" + code
    
    def generate_diff(self, original_code: str, corrected_code: str) -> str:
        """
        Generate a unified diff between original and corrected code.
        
        Args:
            original_code: The original source code
            corrected_code: The corrected code
            
        Returns:
            Unified diff as a string
        """
        original_lines = original_code.splitlines(keepends=True)
        corrected_lines = corrected_code.splitlines(keepends=True)
        
        diff = unified_diff(
            original_lines, corrected_lines,
            fromfile='original', tofile='corrected',
            n=3  # Context lines
        )
        
        return ''.join(diff)

    def highlight_fixed_code(self, original_code: str, corrected_code: str) -> str:
        """
        Generate a highlighted version of the corrected code showing changes.
        
        Args:
            original_code: The original source code
            corrected_code: The corrected code
            
        Returns:
            Highlighted code with annotations showing changes
        """
        from difflib import SequenceMatcher
        
        original_lines = original_code.splitlines()
        corrected_lines = corrected_code.splitlines()
        
        matcher = SequenceMatcher(None, original_lines, corrected_lines)
        result = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Add unchanged lines
                for i in range(i1, i2):
                    result.append(f"  {original_lines[i]}")
            elif tag == 'replace':
                # Show replaced lines
                for i in range(i1, i2):
                    result.append(f"- {original_lines[i]}")
                for j in range(j1, j2):
                    result.append(f"+ {corrected_lines[j]}")
            elif tag == 'delete':
                # Show deleted lines
                for i in range(i1, i2):
                    result.append(f"- {original_lines[i]}")
            elif tag == 'insert':
                # Show inserted lines
                for j in range(j1, j2):
                    result.append(f"+ {corrected_lines[j]}")
        
        return "\n".join(result)


def extract_issues_from_feedback(feedback_items):
    """Extract issues from feedback items to use for code correction."""
    issues = []
    code = feedback_items.get('code', '')
    
    # Process readability issues
    if 'readability' in feedback_items:
        readability = feedback_items['readability']
        if isinstance(readability, str) and 'consider improving' in readability.lower():
            parts = readability.split(': ', 1)
            if len(parts) > 1:
                issues.append({
                    'type': 'readability',
                    'message': parts[1],
                    'line': find_potential_line_for_issue(code, parts[1])
                })
    
    # Process performance issues
    if 'performance' in feedback_items:
        performance = feedback_items['performance']
        if isinstance(performance, list):
            for item in performance:
                if 'nested loops' in item.lower():
                    issues.append({
                        'type': 'performance',
                        'message': item,
                        'line': find_line_with_pattern(code, r'for\s+.+\s+in.+:.*\n\s+for')
                    })
                elif 'recursive' in item.lower():
                    issues.append({
                        'type': 'performance',
                        'message': item,
                        'line': find_recursive_function_line(code)
                    })
                else:
                    issues.append({
                        'type': 'performance',
                        'message': item,
                        'line': find_potential_line_for_issue(code, item)
                    })
        elif isinstance(performance, str) and 'looks good' not in performance.lower():
            issues.append({
                'type': 'performance',
                'message': performance,
                'line': find_potential_line_for_issue(code, performance)
            })
    
    # Process security issues
    if 'security' in feedback_items:
        security = feedback_items['security']
        if isinstance(security, list):
            for item in security:
                # Try to identify the specific function and its line
                func_name = None
                if 'eval' in item.lower():
                    func_name = 'eval'
                elif 'exec' in item.lower():
                    func_name = 'exec'
                elif 'open' in item.lower():
                    func_name = 'open'
                elif 'system' in item.lower():
                    func_name = 'system'
                
                issues.append({
                    'type': 'security',
                    'message': item,
                    'func_name': func_name,
                    'line': find_line_with_pattern(code, fr'\b{func_name}\s*\(') if func_name else find_potential_line_for_issue(code, item)
                })
        elif isinstance(security, str) and 'no' not in security.lower():
            issues.append({
                'type': 'security',
                'message': security,
                'line': find_potential_line_for_issue(code, security)
            })
    
    # Process string concatenation issues
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if '+=' not in line and re.search(r'\w+\s*=\s*\w+\s*\+', line):
            var_match = re.search(r'(\w+)\s*=\s*\1\s*\+', line)
            if var_match:
                # This is a += candidate
                issues.append({
                    'type': 'performance',
                    'message': 'Inefficient string concatenation. Use += instead.',
                    'line': i
                })
    
    # Look for potentially uncaught issues    
    # Check for multiple nested loops
    nested_loops = find_nested_loops(code)
    for loop_start in nested_loops:
        # Check if we already have an issue for this line
        if not any(issue.get('line') == loop_start for issue in issues):
            issues.append({
                'type': 'performance',
                'message': 'Code contains deeply nested loops. Consider refactoring to reduce time complexity.',
                'line': loop_start
            })
    
    # Return unique issues (by line and message)
    unique_issues = []
    seen = set()
    for issue in issues:
        key = (issue.get('line', 0), issue.get('message', ''))
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)
    
    return unique_issues


def find_line_with_pattern(code, pattern):
    """Find the line number containing a regex pattern."""
    import re
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if re.search(pattern, line):
            return i
    
    # If we're looking for a pattern that might span multiple lines
    code_flat = '\n'.join(lines)
    match = re.search(pattern, code_flat)
    if match:
        # Count lines up to the match position
        return code_flat[:match.start()].count('\n') + 1
    
    return 0


def find_recursive_function_line(code):
    """Find the line number of a recursive function definition."""
    import re
    import ast
    
    # Try to parse and find recursive functions
    try:
        tree = ast.parse(code)
        recursive_funcs = set()
        
        class RecursiveVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_func = None
                self.func_lines = {}
            
            def visit_FunctionDef(self, node):
                old_func = self.current_func
                self.current_func = node.name
                self.func_lines[node.name] = node.lineno
                self.generic_visit(node)
                self.current_func = old_func
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == self.current_func:
                    recursive_funcs.add(self.current_func)
                self.generic_visit(node)
        
        visitor = RecursiveVisitor()
        visitor.visit(tree)
        
        if recursive_funcs:
            # Return the line of the first recursive function found
            for func_name in recursive_funcs:
                return visitor.func_lines.get(func_name, 0)
    except:
        pass
    
    # Fallback: simple pattern matching for recursive calls
    lines = code.split('\n')
    func_defs = {}
    
    # First pass: collect function definitions
    for i, line in enumerate(lines, 1):
        match = re.search(r'def\s+(\w+)', line)
        if match:
            func_name = match.group(1)
            func_defs[func_name] = i
    
    # Second pass: look for recursive calls
    for func_name, line_no in func_defs.items():
        # Search for function calls to itself in the next 20 lines (approximate function body)
        for i in range(line_no, min(line_no + 20, len(lines) + 1)):
            if re.search(r'\b' + re.escape(func_name) + r'\s*\(', lines[i-1]):
                if i != line_no:  # Not the definition line itself
                    return line_no  # Return the function definition line
    
    return 0


def find_nested_loops(code):
    """Find all lines containing the start of nested loops."""
    import re
    import ast
    
    nested_loop_lines = []
    
    # Try to parse the code with ast
    try:
        tree = ast.parse(code)
        
        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.loop_stack = []
                self.nested_loops = []
            
            def visit_For(self, node):
                self.loop_stack.append(node)
                if len(self.loop_stack) >= 2:
                    self.nested_loops.append(self.loop_stack[0].lineno)
                self.generic_visit(node)
                self.loop_stack.pop()
            
            def visit_While(self, node):
                self.loop_stack.append(node)
                if len(self.loop_stack) >= 2:
                    self.nested_loops.append(self.loop_stack[0].lineno)
                self.generic_visit(node)
                self.loop_stack.pop()
        
        visitor = LoopVisitor()
        visitor.visit(tree)
        nested_loop_lines = list(set(visitor.nested_loops))
    except:
        # Fallback: simple regex pattern matching
        lines = code.split('\n')
        indent_stack = []
        
        for i, line in enumerate(lines, 1):
            indent = len(line) - len(line.lstrip())
            
            if re.search(r'\bfor\b|\bwhile\b', line.lstrip()):
                # This is a loop header
                while indent_stack and indent_stack[-1][0] >= indent:
                    indent_stack.pop()
                
                if indent_stack:  # If there's already a loop in the stack, this is nested
                    parent_line = indent_stack[0][1]  # Get the outermost loop
                    if parent_line not in nested_loop_lines:
                        nested_loop_lines.append(parent_line)
                
                indent_stack.append((indent, i))
    
    return sorted(nested_loop_lines)


def find_potential_line_for_issue(code, issue_text):
    """Try to find the most likely line number for an issue based on its description."""
    import re
    
    issue_lower = issue_text.lower()
    lines = code.split('\n')
    
    # Check for specific issue types
    if 'indent' in issue_lower:
        # Look for inconsistent indentation
        prev_indent = -1
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if prev_indent >= 0 and indent > prev_indent and indent % 4 != 0:
                return i
            prev_indent = indent
    
    elif 'docstring' in issue_lower:
        # Look for functions without docstrings
        for i, line in enumerate(lines, 1):
            if re.search(r'def\s+\w+\s*\(', line):
                if i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line.startswith('"""') and not next_line.startswith("'''"):
                        return i
    
    elif 'variable names' in issue_lower or 'naming convention' in issue_lower:
        # Look for non-standard variable names
        for i, line in enumerate(lines, 1):
            if '=' in line and not re.search(r'^\s*def', line):
                var_match = re.search(r'^\s*([A-Z][a-z0-9_]*|[a-z][A-Z])\s*=', line)
                if var_match:
                    return i
    
    # Generic fallbacks based on keywords in the issue
    for keyword in ['eval', 'exec', 'open', 'system', 'shell', 'recursion', 'recursive', 'nested loop']:
        if keyword in issue_lower:
            for i, line in enumerate(lines, 1):
                if keyword in line.lower():
                    return i
    
    # Last resort: return first line
    return 1


# Example usage
if __name__ == "__main__":
    # Test code with various issues
    code = """
def calculate_fibonacci(n):
    # This function calculates Fibonacci numbers inefficiently
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    # Using recursion which is inefficient for Fibonacci
    def fib(n):
        if n <= 1:
            return n
        else:
            return fib(n-1) + fib(n-2)
    
    return [fib(i) for i in range(n)]

def process_data(input_file):
    # Opening file without proper error handling
    f = open(input_file, 'r')
    data = f.readlines()
    
    # Inefficient string concatenation in a loop
    result = ""
    for line in data:
        result = result + line.strip() + ";"
    
    # Security issue: eval on input data
    processed = eval("len(data) + 100")
    
    return result, processed
"""
    
    # Create simulated issues
    issues = [
        {
            "type": "security",
            "message": "Potential security issue: open - Ensure proper file handling and permissions checking.",
            "func_name": "open",
            "line": 17
        },
        {
            "type": "security",
            "message": "Potential security issue: eval - Code injection risk. Consider safer alternatives.",
            "func_name": "eval",
            "line": 25
        },
        {
            "type": "performance",
            "message": "Inefficient string concatenation in loop. Use string join or append to list.",
            "line": 22
        },
        {
            "type": "performance",
            "message": "Recursive function detected. Ensure it has a base case to avoid infinite recursion.",
            "line": 9
        }
    ]
    
    corrector = CodeCorrector()
    corrected_code = corrector.correct_code(code, issues)
    
    print("=== Original Code ===")
    print(code)
    print("\n=== Corrected Code ===")
    print(corrected_code)
    print("\n=== Highlighted Differences ===")
    print(corrector.highlight_fixed_code(code, corrected_code))