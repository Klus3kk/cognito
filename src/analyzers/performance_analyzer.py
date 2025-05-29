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
    """
    Analyze memory usage patterns through static code analysis.
    
    Args:
        code_snippet (str): Python code to analyze
        
    Returns:
        str: Memory usage analysis and suggestions
    """
    import ast
    import re
    
    try:
        # Parse the code into AST
        tree = ast.parse(code_snippet)
        
        # Simple pattern detection
        memory_issues = []
        
        # 1. Large data structures
        large_lists = len(re.findall(r'\[[^\]]{30,}\]', code_snippet))
        large_dicts = len(re.findall(r'\{[^}]{30,}\}', code_snippet))
        
        if large_lists + large_dicts > 1:
            memory_issues.append("large data structures detected")
        
        # 2. String concatenation in loops
        lines = code_snippet.split('\n')
        concat_in_loop = False
        for i, line in enumerate(lines):
            if 'for ' in line and i + 1 < len(lines):
                next_lines = lines[i+1:i+4]  # Check next few lines
                for next_line in next_lines:
                    if '= ' in next_line and ' + ' in next_line and ('str(' in next_line or '"' in next_line or "'" in next_line):
                        concat_in_loop = True
                        break
        
        if concat_in_loop:
            memory_issues.append("string concatenation in loop - use join() instead")
        
        # 3. Nested loops (O(n²) or higher complexity)
        nested_loops = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check if this loop contains another loop
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.For, ast.While)):
                        nested_loops += 1
                        break
        
        if nested_loops > 0:
            memory_issues.append("nested loops may create large temporary data")
        
        # 4. Recursive functions
        functions = []
        recursive_funcs = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        for func_name in functions:
            if func_name in code_snippet.count(f'{func_name}(') > 1:  # Simple heuristic
                recursive_funcs += 1
        
        if recursive_funcs > 0:
            memory_issues.append("recursive functions may cause stack overflow")
        
        # 5. Check for efficient patterns
        efficient_patterns = []
        if 'yield ' in code_snippet:
            efficient_patterns.append("uses memory-efficient generators")
        if '[' in code_snippet and 'for ' in code_snippet and ' in ' in code_snippet:
            if code_snippet.count('[') < code_snippet.count('for'):  # List comprehension heuristic
                efficient_patterns.append("uses efficient list comprehensions")
        
        # Estimate memory complexity
        lines_count = len(code_snippet.split('\n'))
        if nested_loops > 0 or large_lists + large_dicts > 2:
            complexity = "High complexity"
        elif lines_count > 50 or len(memory_issues) > 1:
            complexity = "Medium complexity"
        else:
            complexity = "Low complexity"
        
        # Format result
        if memory_issues:
            issue_str = '; '.join(memory_issues[:2])  # Show top 2 issues
            return f"Memory usage: {complexity} - {issue_str}"
        elif efficient_patterns:
            pattern_str = ', '.join(efficient_patterns)
            return f"Memory usage: {complexity} - {pattern_str}"
        else:
            return f"Memory usage: {complexity} - no major concerns detected"
            
    except Exception as e:
        # Fallback to simple line count estimation
        lines = len(code_snippet.split('\n'))
        if lines > 100:
            return "Memory usage: High complexity - large codebase"
        elif lines > 50:
            return "Memory usage: Medium complexity - moderate size"
        else:
            return "Memory usage: Low complexity - small codebase"