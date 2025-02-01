import ast
import memory_profiler

class ComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.nested_loops = 0
        self.recursion_count = 0
        self.function_defs = set()

    def visit_For(self, node):
        self.nested_loops += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.nested_loops += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.function_defs.add(node.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check if function calls itself (recursion detection)
        if isinstance(node.func, ast.Name) and node.func.id in self.function_defs:
            self.recursion_count += 1
        self.generic_visit(node)

def analyze_complexity(code_snippet):
    try:
        tree = ast.parse(code_snippet)
        analyzer = ComplexityAnalyzer()
        analyzer.visit(tree)

        suggestions = []
        if analyzer.nested_loops > 2:
            suggestions.append("Code contains deeply nested loops. Consider refactoring to reduce time complexity.")

        if analyzer.recursion_count > 0:
            suggestions.append("Recursive function detected. Ensure it has a base case to avoid infinite recursion.")

        if not suggestions:
            return "Code complexity looks good."

        return suggestions
    except Exception as e:
        return [f"Error analyzing complexity: {str(e)}"]

def analyze_memory_usage(code_snippet):
    exec_globals = {}
    memory_before = memory_profiler.memory_usage()[0]

    try:
        exec(code_snippet, exec_globals)
        memory_after = memory_profiler.memory_usage()[0]
        memory_used = memory_after - memory_before
        return f"Memory usage: {memory_used:.2f} MB"
    except Exception as e:
        return f"Error analyzing memory: {str(e)}"