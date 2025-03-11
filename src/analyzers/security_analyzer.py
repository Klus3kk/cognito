import ast
import re

class SecurityVisitor(ast.NodeVisitor):
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
        
        # OWASP Top 10 for Python patterns
        self.sql_injection_pattern = re.compile(r'.*execute\s*\(\s*["\']SELECT.*%s.*["\']\s*,')
        self.path_traversal_pattern = re.compile(r'.*open\s*\(\s*.*\+\s*.*\)')
    
    def visit_Call(self, node):
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.dangerous_functions:
                self.security_issues.append(f"Potential security issue: {func_name} - {self.dangerous_functions[func_name]}")
        
        # Check for method calls on objects
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                method_name = f"{node.func.value.id}.{node.func.attr}"
                if method_name in self.dangerous_functions:
                    self.security_issues.append(f"Potential security issue: {method_name} - {self.dangerous_functions[method_name]}")
            
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Convert to source code for regex patterns
        source = ast.unparse(node)
        
        # Check for SQL injection patterns
        if self.sql_injection_pattern.match(source):
            self.security_issues.append("Potential SQL Injection detected. Use parameterized queries instead.")
        
        # Check for path traversal
        if self.path_traversal_pattern.match(source):
            self.security_issues.append("Potential Path Traversal vulnerability. Validate and sanitize file paths.")
            
        self.generic_visit(node)


def analyze_security(code_snippet):
    """
    Analyze code for security vulnerabilities based on OWASP guidelines.
    
    Args:
        code_snippet (str): Python code to analyze
    
    Returns:
        list: Security issues found in the code
    """
    try:
        tree = ast.parse(code_snippet)
        visitor = SecurityVisitor()
        visitor.visit(tree)
        
        if not visitor.security_issues:
            return ["No common security issues detected."]
        
        return visitor.security_issues
    except Exception as e:
        return [f"Error analyzing security: {str(e)}"]


def generate_security_suggestion(issues):
    """
    Generate actionable suggestions based on security issues
    
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