import ast
import memory_profiler

class EnhancedComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.nested_loops = []
        self.recursion_info = []
        self.function_defs = set()
        self.current_function = None
        self.line_info = {}

    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        self.function_defs.add(node.name)
        
        # Store function line info
        self.line_info[node.name] = {
            'line': node.lineno,
            'end_line': getattr(node, 'end_lineno', None)
        }
        
        self.generic_visit(node)
        self.current_function = old_function

    def visit_For(self, node):
        if hasattr(node, 'parent') and isinstance(node.parent, ast.For):
            # This is a nested loop
            parent_loop = node.parent
            if hasattr(parent_loop, 'parent') and isinstance(parent_loop.parent, ast.For):
                # This is a deeply nested loop (3+ levels)
                self.nested_loops.append({
                    'line': parent_loop.parent.lineno,
                    'message': "Code contains deeply nested loops. Consider refactoring to reduce time complexity."
                })
        
        # Set parent reference for child nodes
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
        
        self.generic_visit(node)

    def visit_While(self, node):
        if hasattr(node, 'parent') and isinstance(node.parent, (ast.For, ast.While)):
            # This is a nested loop
            parent_loop = node.parent
            if hasattr(parent_loop, 'parent') and isinstance(parent_loop.parent, (ast.For, ast.While)):
                # This is a deeply nested loop (3+ levels)
                self.nested_loops.append({
                    'line': parent_loop.parent.lineno,
                    'message': "Code contains deeply nested loops. Consider refactoring to reduce time complexity."
                })
        
        # Set parent reference for child nodes
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
        
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check if function calls itself (recursion detection)
        if isinstance(node.func, ast.Name) and node.func.id in self.function_defs:
            func_name = node.func.id
            # Only report recursion if the function calls itself directly
            if func_name == self.current_function:
                # Check if we've already recorded this recursive function
                if not any(r.get('func_name') == func_name for r in self.recursion_info):
                    self.recursion_info.append({
                        'line': self.line_info.get(func_name, {}).get('line', node.lineno),
                        'func_name': func_name,
                        'message': "Recursive function detected. Ensure it has a base case to avoid infinite recursion."
                    })
        
        self.generic_visit(node)


def analyze_complexity_with_lines(code_snippet):
    """
    Analyze code complexity with line number information.
    
    Args:
        code_snippet (str): Python code to analyze
        
    Returns:
        list: Analysis results with line numbers
    """
    try:
        tree = ast.parse(code_snippet)
        
        # Set parent references to improve context
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                setattr(child, 'parent', node)
        
        analyzer = EnhancedComplexityAnalyzer()
        analyzer.visit(tree)

        issues = []
        
        # Add nested loop issues
        for loop_info in analyzer.nested_loops:
            issues.append(loop_info)

        # Add recursion issues
        for rec_info in analyzer.recursion_info:
            issues.append(rec_info)

        if not issues:
            issues.append({
                'line': 0,
                'message': "Code complexity looks good."
            })

        return issues
    except Exception as e:
        return [{
            'line': 0,
            'message': f"Error analyzing complexity: {str(e)}"
        }]


def analyze_complexity(code_snippet):
    """
    Analyze complexity (compatibility wrapper for existing code).
    
    Args:
        code_snippet (str): Python code to analyze
        
    Returns:
        list or str: Complexity analysis results in the original format
    """
    detailed_results = analyze_complexity_with_lines(code_snippet)
    
    # Convert to the format expected by the original function
    simplified_results = []
    for issue in detailed_results:
        message = issue.get('message')
        if message:
            simplified_results.append(message)
    
    # Handle empty results
    if not simplified_results:
        return "Code complexity looks good."
    
    # Check if all results are positive
    if all("looks good" in msg.lower() for msg in simplified_results):
        return "Code complexity looks good."
    
    return simplified_results


def analyze_memory_usage(code_snippet):
    """Analyze memory usage (original function, unchanged)"""
    exec_globals = {}
    memory_before = memory_profiler.memory_usage()[0]

    try:
        exec(code_snippet, exec_globals)
        memory_after = memory_profiler.memory_usage()[0]
        memory_used = memory_after - memory_before
        return f"Memory usage: {memory_used:.2f} MB"
    except Exception as e:
        return f"Error analyzing memory: {str(e)}"