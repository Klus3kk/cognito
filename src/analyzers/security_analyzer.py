"""
Security analyzer that identifies issues with line numbers
"""
import ast
import re
from typing import List, Dict, Any, Tuple, Optional

class SecurityVisitorWithLineNumbers(ast.NodeVisitor):
    def __init__(self):
        self.security_issues = []
        self.dangerous_functions = {
            'eval': 'Code injection risk. Consider safer alternatives.',
            'exec': 'Code injection risk. Consider safer alternatives.',
            'os.system': 'Command injection risk. Use subprocess with shell=False instead.',
            'subprocess.call': 'Check if shell=True is used. If so, command injection risk.',
            'pickle.load': 'Deserialization vulnerability. Avoid loading untrusted data.',
            'yaml.load': 'Use yaml.safe_load instead to prevent execution of arbitrary code.',
            'open': 'Ensure proper file handling and permissions checking.'
        }
        
    def visit_Call(self, node):
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.dangerous_functions:
                self.security_issues.append({
                    "line": node.lineno,
                    "col": node.col_offset,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "end_col": getattr(node, "end_col_offset", node.col_offset + len(func_name)),
                    "func_name": func_name,
                    "message": f"Potential security issue: {func_name} - {self.dangerous_functions[func_name]}",
                    "suggestion": self._generate_fix(func_name, node)
                })
        
        # Check for method calls on objects
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                method_name = f"{node.func.value.id}.{node.func.attr}"
                if method_name in self.dangerous_functions:
                    self.security_issues.append({
                        "line": node.lineno,
                        "col": node.col_offset,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                        "end_col": getattr(node, "end_col_offset", node.col_offset + len(method_name)),
                        "func_name": method_name,
                        "message": f"Potential security issue: {method_name} - {self.dangerous_functions[method_name]}",
                        "suggestion": self._generate_fix(method_name, node)
                    })
            
        self.generic_visit(node)
    
    def _generate_fix(self, func_name: str, node: ast.Call) -> Optional[str]:
        """Generate a suggested fix for the security issue"""
        if func_name == 'eval':
            # Try to safely convert eval to a safer alternative
            try:
                if isinstance(node.args[0], ast.Str):
                    # Simple arithmetic expressions can be replaced with ast.literal_eval
                    return f"import ast\nast.literal_eval({ast.unparse(node.args[0])})"
                else:
                    return "import ast\nast.literal_eval(...) # Replace with appropriate expression"
            except (IndexError, AttributeError):
                return "# Replace eval with a safer alternative"
        
        elif func_name == 'open':
            # Suggest using a context manager
            try:
                args_str = ", ".join(ast.unparse(arg) for arg in node.args)
                kwargs_str = ", ".join(f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords)
                params = ", ".join(filter(None, [args_str, kwargs_str]))
                
                # Check if the open result is assigned to a variable
                parent = getattr(node, 'parent', None)
                var_name = None
                next_var_name = None
                
                # Try to find variable name in assignment
                if isinstance(parent, ast.Assign):
                    if len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
                        var_name = parent.targets[0].id
                        
                        # Try to find the next statement that uses this variable
                        parent_parent = getattr(parent, 'parent', None)
                        if isinstance(parent_parent, ast.Module) and parent_parent.body:
                            # Find the index of the current assignment in the module body
                            idx = -1
                            for i, stmt in enumerate(parent_parent.body):
                                if stmt is parent:
                                    idx = i
                                    break
                            
                            # Look for the next statement that might use this variable
                            if idx >= 0 and idx + 1 < len(parent_parent.body):
                                next_stmt = parent_parent.body[idx + 1]
                                if isinstance(next_stmt, ast.Assign):
                                    # Check if the next statement assigns to a variable using our file handle
                                    if isinstance(next_stmt.value, ast.Call) and \
                                    isinstance(next_stmt.value.func, ast.Attribute) and \
                                    isinstance(next_stmt.value.func.value, ast.Name) and \
                                    next_stmt.value.func.value.id == var_name and \
                                    next_stmt.value.func.attr == 'readlines' and \
                                    len(next_stmt.targets) == 1 and \
                                    isinstance(next_stmt.targets[0], ast.Name):
                                        next_var_name = next_stmt.targets[0].id
                
                # Generate appropriate fix based on the variable names found
                if next_var_name:
                    # If we found a pattern like: f = open(...) followed by data = f.readlines()
                    return f"""
    try:
        with open({params}) as file_handler:
            {next_var_name} = file_handler.readlines()  # Direct assignment to data variable
    except FileNotFoundError:
        print(f"File not found: {{{args_str.split(',')[0]}}}")  # Handle error appropriately
        {next_var_name} = []  # Provide a default value
    except IOError as e:
        print(f"I/O error: {{e}}")  # Handle error appropriately
        {next_var_name} = []  # Provide a default value
    """
                elif var_name:
                    # If we just found a variable assignment but don't know what it's used for
                    return f"""
    try:
        with open({params}) as {var_name}:  # Using the same variable name as a context manager
            # Your file operations here (e.g. content = {var_name}.read())
            content = {var_name}.read()  # Example operation
    except FileNotFoundError:
        print(f"File not found: {{{args_str.split(',')[0]}}}")  # Handle error appropriately
        content = ""  # Provide a default value
    except IOError as e:
        print(f"I/O error: {{e}}")  # Handle error appropriately
        content = ""  # Provide a default value
    """
                else:
                    # Generic case for when we don't know the variable name
                    return f"""
    try:
        with open({params}) as file_handler:
            # Your file operations here
            data = file_handler.readlines()  # Example operation
    except FileNotFoundError:
        print(f"File not found: {{{args_str.split(',')[0]}}}")  # Handle error appropriately
        data = []  # Provide a default value
    except IOError as e:
        print(f"I/O error: {{e}}")  # Handle error appropriately
        data = []  # Provide a default value
    """
            except Exception as e:
                # Fallback to generic suggestion if parsing fails
                return """
    try:
        with open(...) as file_handler:
            # Your file operations here
            data = file_handler.read()  # Example operation
    except FileNotFoundError:
        # Handle file not found
        data = ""  # Provide a default value
    except IOError:
        # Handle other IO errors
        data = ""  # Provide a default value
    """
        
        elif func_name == 'os.system':
            return """
    import subprocess
    subprocess.run([...], shell=False, check=True)
    """
        
        return None  # No specific fix suggestion

def analyze_security(code_snippet):
    """
    Wrapper for backward compatibility with Cognito.
    Analyzes code for security vulnerabilities based on OWASP guidelines.
    
    Args:
        code_snippet (str): Python code to analyze
    
    Returns:
        list: Security issues found in the code (simplified format for compatibility)
    """
    # Call the detailed analyzer
    detailed_issues = analyze_security_detailed(code_snippet)
    
    # Convert to the format expected by the original function
    simplified_issues = []
    
    for issue in detailed_issues:
        message = issue.get("message", "")
        if message and "No common security issues detected" not in message:
            simplified_issues.append(message)
    
    # If no issues were found, return the standard "no issues" message
    if not simplified_issues:
        return ["No common security issues detected."]
    
    return simplified_issues


def generate_security_suggestion(issues):
    """
    Generate actionable suggestions based on security issues.
    Maintained for backward compatibility.
    
    Args:
        issues (list): List of security issues found
    
    Returns:
        str: Formatted suggestion with OWASP reference
    """
    if len(issues) == 1 and issues[0] == "No common security issues detected.":
        return "Security Analysis: Code passes basic OWASP security checks."
    
    suggestion = "Security Analysis: The following issues were found:\n"
    for issue in issues:
        suggestion += f"- {issue}\n"
    
    suggestion += "\nRecommendation: Review OWASP Top 10 guidelines for secure coding practices."
    return suggestion

def analyze_security_detailed(code_snippet: str) -> List[Dict[str, Any]]:
    """
    Analyze code for security vulnerabilities with line information.
    
    Args:
        code_snippet (str): Python code to analyze
    
    Returns:
        list: Security issues found in the code with line numbers
    """
    try:
        tree = ast.parse(code_snippet)
        visitor = SecurityVisitorWithLineNumbers()
        visitor.visit(tree)
        
        if not visitor.security_issues:
            return [{
                "line": 0,
                "col": 0,
                "message": "No common security issues detected.",
                "suggestion": None
            }]
        
        return visitor.security_issues
    except Exception as e:
        return [{
            "line": 0,
            "col": 0,
            "message": f"Error analyzing security: {str(e)}",
            "suggestion": None
        }]

def format_code_with_highlights(code_snippet: str, issues: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Format code with highlighted issues.
    
    Args:
        code_snippet (str): The original code
        issues (List[Dict[str, Any]]): List of issues with line information
        
    Returns:
        Tuple[str, List[Dict[str, Any]]]: Formatted code and enriched issues
    """
    lines = code_snippet.split('\n')
    issue_lines = {}
    
    # Group issues by line
    for issue in issues:
        line_num = issue.get("line", 0)
        if line_num > 0 and line_num <= len(lines):
            if line_num not in issue_lines:
                issue_lines[line_num] = []
            issue_lines[line_num].append(issue)
    
    # Format the code with issues highlighted
    formatted_lines = []
    for i, line in enumerate(lines, 1):
        if i in issue_lines:
            # Add the original line
            formatted_lines.append(f"  {line}")
            
            # Add indicators pointing to the issue
            indicators = [" " for _ in range(len(line) + 2)]
            for issue in issue_lines[i]:
                col = issue.get("col", 0)
                if col >= 0:
                    # Adjust for the 2-space prefix
                    start_col = col + 2
                    end_col = min(issue.get("end_col", col + 1) + 2, len(indicators))
                    for j in range(start_col, end_col):
                        indicators[j] = "^"
            
            formatted_lines.append("".join(indicators))
            
            # Add the issue message
            for issue in issue_lines[i]:
                formatted_lines.append(f"  # Error: {issue['message']}")
        else:
            formatted_lines.append(f"  {line}")
    
    return "\n".join(formatted_lines), issues

def create_fixed_code(code_snippet: str, issues: List[Dict[str, Any]]) -> str:
    """
    Generate fixed code based on the suggestions.
    
    Args:
        code_snippet (str): The original code
        issues (List[Dict[str, Any]]): List of issues with suggestions
        
    Returns:
        str: The fixed code
    """
    # This is a simplified version - a real implementation would be more complex
    lines = code_snippet.split('\n')
    fixed_lines = lines.copy()
    
    # Apply fixes from bottom to top to avoid line number changes
    sorted_issues = sorted(
        [issue for issue in issues if issue.get("suggestion")], 
        key=lambda x: x.get("line", 0), 
        reverse=True
    )
    
    for issue in sorted_issues:
        line_num = issue.get("line", 0)
        if line_num > 0 and line_num <= len(fixed_lines):
            suggestion = issue.get("suggestion")
            if suggestion:
                # Simple replacement for now
                fixed_lines[line_num-1] = f"# FIXED: {fixed_lines[line_num-1]}"
                # Add the suggestion indented properly
                indent = len(fixed_lines[line_num-1]) - len(fixed_lines[line_num-1].lstrip())
                indent_str = " " * indent
                suggestion_lines = suggestion.strip().split('\n')
                # Insert after the commented-out line
                for i, sug_line in enumerate(suggestion_lines):
                    fixed_lines.insert(line_num + i, f"{indent_str}{sug_line}")
    
    return "\n".join(fixed_lines)

# Sample usage
if __name__ == "__main__":
    code = """
def process_data(input_file):
    # Opening file without proper error handling
    f = open(input_file, 'r')
    data = f.readlines()
    
    # Security issue: eval on input data
    processed = eval("len(data) + 100")
    
    return result, processed
"""
    
    issues = analyze_security_detailed(code)
    formatted_code, enriched_issues = format_code_with_highlights(code, issues)
    fixed_code = create_fixed_code(code, issues)
    
    print("=== Original Code with Issues ===")
    print(formatted_code)
    print("\n=== Suggested Fixed Code ===")
    print(fixed_code)