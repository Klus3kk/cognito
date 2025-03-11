import ast
import re
from collections import defaultdict

def analyze_imports(code):
    """
    Analyze imports for potential issues and optimizations.
    
    Args:
        code (str): Python code to analyze
        
    Returns:
        list: Suggestions regarding imports
    """
    suggestions = []
    try:
        tree = ast.parse(code)
        import_nodes = []
        
        # Collect all import nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)
        
        # Check for imports not at the top of the file
        non_top_imports = []
        import_lines = set()
        
        for node in import_nodes:
            if hasattr(node, 'lineno'):
                import_lines.add(node.lineno)
                
        # Sort line numbers to identify gaps
        sorted_lines = sorted(import_lines)
        if sorted_lines:
            for i in range(1, len(sorted_lines)):
                if sorted_lines[i] - sorted_lines[i-1] > 1:
                    non_top_imports.append(sorted_lines[i])
        
        if non_top_imports:
            suggestions.append(
                f"Imports found scattered throughout the code (lines: {non_top_imports}). "
                f"Consider placing all imports at the top of the file."
            )
        
        # Check for unused imports (basic check, not perfect)
        import_names = set()
        for node in import_nodes:
            if isinstance(node, ast.Import):
                for name in node.names:
                    import_names.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    if name.name != '*':
                        if node.module:
                            import_names.add(f"{node.module}.{name.name}")
                        else:
                            import_names.add(name.name)
        
        # Simple usage check (not comprehensive)
        code_text = code.lower()
        potentially_unused = []
        
        for name in import_names:
            base_name = name.split('.')[-1].lower()
            if base_name not in code_text[code_text.find('import'):]:
                potentially_unused.append(name)
        
        if potentially_unused:
            suggestions.append(
                f"Potentially unused imports detected: {', '.join(potentially_unused)}. "
                f"Consider removing them to clean up the code."
            )
        
        # If no issues found
        if not suggestions:
            suggestions.append("Import structure looks good.")
            
        return suggestions
    
    except Exception as e:
        return [f"Error analyzing imports: {str(e)}"]


def get_maintainability_index(code):
    """
    Calculate the maintainability index for the code.
    
    The maintainability index is a software metric that represents how maintainable the code is.
    Values range from 0 to 100, with higher values meaning better maintainability.
    
    Args:
        code (str): Python code to analyze
        
    Returns:
        float: Maintainability index value
    """
    import math
    
    try:
        # Count lines of code (excluding empty lines and comments)
        loc = 0
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                loc += 1
        
        if loc == 0:
            return 100.0  # Empty code is trivially maintainable
        
        # Count tokens for Halstead volume (simplified)
        operators = set()
        operands = set()
        
        tree = ast.parse(code)
        
        # Collect operators
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                operators.add(type(node.op).__name__)
            elif isinstance(node, ast.UnaryOp):
                operators.add(type(node.op).__name__)
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    operators.add(type(op).__name__)
        
        # Collect operands (variable names and literals)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                operands.add(node.id)
            elif isinstance(node, (ast.Str, ast.Num, ast.Constant)):
                # Handle different Python versions
                if hasattr(node, 's'):
                    operands.add(str(node.s))
                elif hasattr(node, 'n'):
                    operands.add(str(node.n))
                elif hasattr(node, 'value'):
                    operands.add(str(node.value))
        
        # Calculate Halstead metrics
        n1 = len(operators)
        n2 = len(operands)
        N1 = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.UnaryOp)) or 
                (isinstance(node, ast.Compare) and node.ops))
        N2 = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Name, ast.Str, ast.Num, ast.Constant)))
        
        # Prevent division by zero
        n1 = max(n1, 1)
        n2 = max(n2, 1)
        N1 = max(N1, 1)
        N2 = max(N2, 1)
        
        # Calculate Halstead volume
        volume = (N1 + N2) * math.log2(n1 + n2)
        
        # Calculate cyclomatic complexity (simplified)
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
        
        # Calculate maintainability index
        # MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        halstead_volume = 0 if volume <= 0 else math.log(volume)
        maintainability = 171 - 5.2 * halstead_volume - 0.23 * complexity - 16.2 * math.log(loc)
        
        # Normalize to 0-100 scale
        maintainability = max(0, min(100, maintainability))
        
        return round(maintainability, 2)
    
    except Exception:
        # Return a default value if analysis fails
        return 50.0


def get_python_analysis_summary(analysis_results):
    """
    Generate a comprehensive summary of the Python code analysis.
    
    Args:
        analysis_results (dict): Results from analyze_python
        
    Returns:
        dict: Summary with key metrics and recommendations
    """
    style_issues = len([issue for issue in analysis_results.get('style', {}).get('suggestions', []) 
                       if 'No style issues detected' not in issue])
    
    anti_patterns = len([pattern for pattern in analysis_results.get('best_practices', []) 
                        if 'No common anti-patterns detected' not in pattern])
    
    # Get a maintainability rating based on the maintainability index
    maintainability_index = analysis_results.get('maintainability_index', 50)
    if maintainability_index >= 85:
        maintainability_rating = "Excellent"
    elif maintainability_index >= 65:
        maintainability_rating = "Good"
    elif maintainability_index >= 40:
        maintainability_rating = "Fair"
    else:
        maintainability_rating = "Poor"
    
    # Generate priority recommendations
    recommendations = []
    
    # Add style recommendations
    for suggestion in analysis_results.get('style', {}).get('suggestions', [])[:2]:
        if 'No style issues detected' not in suggestion:
            recommendations.append(suggestion)
    
    # Add anti-pattern recommendations
    for pattern in analysis_results.get('best_practices', [])[:2]:
        if 'No common anti-patterns detected' not in pattern:
            recommendations.append(pattern)
    
    # Get complexity metrics
    avg_complexity = analysis_results.get('style', {}).get('stats', {}).get('avg_complexity', 0)
    max_line_length = analysis_results.get('style', {}).get('stats', {}).get('max_line_length', 0)
    
    summary = {
        'style_issues': style_issues,
        'anti_patterns': anti_patterns,
        'maintainability': {
            'index': maintainability_index,
            'rating': maintainability_rating
        },
        'complexity': {
            'average': round(avg_complexity, 1) if avg_complexity else 0,
            'assessment': 'High' if avg_complexity > 7 else 'Moderate' if avg_complexity > 4 else 'Low'
        },
        'readability': {
            'max_line_length': max_line_length,
            'assessment': 'Poor' if max_line_length > 100 else 'Fair' if max_line_length > 79 else 'Good'
        },
        'priority_recommendations': recommendations[:3]  # Top 3 recommendations
    }
    
    return summary

def analyze_python(code):
    """
    Main entry point for comprehensive Python code analysis.
    
    Args:
        code (str): Python code to analyze
        
    Returns:
        dict: Analysis results with categories and suggestions
    """
    results = {
        'style': {},
        'complexity': {},
        'security': {},
        'best_practices': {}
    }
    
    # Style analysis
    style_results = analyze_python_style(code)
    style_suggestions = generate_style_suggestions(style_results)
    results['style'] = {
        'issues': style_results['issues'],
        'stats': style_results['stats'],
        'suggestions': style_suggestions
    }
    
    # Detect anti-patterns and code smells
    anti_patterns = detect_python_anti_patterns(code)
    results['best_practices'] = anti_patterns
    
    return results


def detect_python_anti_patterns(code):
    """
    Detect common Python anti-patterns and code smells.
    
    Args:
        code (str): Python code to analyze
        
    Returns:
        dict: Anti-patterns found with suggestions
    """
    anti_patterns = []
    
    try:
        tree = ast.parse(code)
        
        # Check for mutable default arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.defaults:
                    if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                        anti_patterns.append(
                            f"Function '{node.name}' uses mutable default argument. "
                            f"This can lead to unexpected behavior. Use None instead and initialize in the function body."
                        )
        
        # Check for wildcard imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.names[0].name == '*':
                anti_patterns.append(
                    f"Wildcard import found (from {node.module} import *). "
                    f"This pollutes the namespace and makes it hard to track where names are defined."
                )
        
        # Check for bare excepts
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                anti_patterns.append(
                    "Bare 'except:' found. This catches all exceptions including KeyboardInterrupt and SystemExit. "
                    "Use 'except Exception:' or specify particular exceptions."
                )
                
        # Check for repeated string literals
        string_literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Str) and len(node.s) > 5:
                if node.s in string_literals:
                    string_literals[node.s] += 1
                else:
                    string_literals[node.s] = 1
                    
        for string, count in string_literals.items():
            if count > 3 and len(string) > 10:  # Only flag if it's a substantial string that repeats
                preview = string[:20] + "..." if len(string) > 20 else string
                anti_patterns.append(
                    f"String literal '{preview}' is repeated {count} times. "
                    f"Consider defining it as a constant."
                )
        
        # If no anti-patterns found
        if not anti_patterns:
            anti_patterns.append("No common anti-patterns detected.")
            
        return anti_patterns
    
    except Exception as e:
        return [f"Error analyzing anti-patterns: {str(e)}"]


class PythonStyleVisitor(ast.NodeVisitor):
    """AST visitor to analyze Python coding style."""
    
    def __init__(self):
        self.issues = []
        self.function_names = []
        self.variable_names = []
        self.class_names = []
        self.import_count = 0
        self.docstring_count = 0
        self.functions_with_docstrings = 0
        self.classes_with_docstrings = 0
        self.function_count = 0
        self.class_count = 0
        self.line_lengths = []
        self.complexity_scores = defaultdict(int)
        
    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.function_names.append(node.name)
        
        # Check function naming convention (should be snake_case)
        if not re.match(r'^[a-z][a-z0-9_]*$', node.name):
            self.issues.append(f"Function '{node.name}' doesn't follow snake_case naming convention")
        
        # Check docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            self.functions_with_docstrings += 1
            self.docstring_count += 1
        
        # Calculate cyclomatic complexity
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
        
        self.complexity_scores[node.name] = complexity
        if complexity > 10:
            self.issues.append(f"Function '{node.name}' has high cyclomatic complexity ({complexity})")
        
        # Continue with other nodes
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.class_count += 1
        self.class_names.append(node.name)
        
        # Check class naming convention (should be PascalCase)
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
            self.issues.append(f"Class '{node.name}' doesn't follow PascalCase naming convention")
        
        # Check docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            self.classes_with_docstrings += 1
            self.docstring_count += 1
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variable_names.append(target.id)
                
                # Check variable naming convention (should be snake_case)
                if not re.match(r'^[a-z][a-z0-9_]*$', target.id) and not target.id.isupper():
                    self.issues.append(f"Variable '{target.id}' doesn't follow snake_case naming convention")
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        self.import_count += len(node.names)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        self.import_count += len(node.names)
        self.generic_visit(node)


def analyze_python_style(code):
    """
    Analyze Python code for style issues.
    
    Args:
        code (str): Python code to analyze
    
    Returns:
        dict: Analysis results
    """
    try:
        tree = ast.parse(code)
        visitor = PythonStyleVisitor()
        visitor.visit(tree)
        
        # Calculate line length statistics
        lines = code.split('\n')
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                visitor.line_lengths.append(len(line))
        
        # Calculate docstring coverage
        function_docstring_coverage = (visitor.functions_with_docstrings / visitor.function_count * 100 
                                      if visitor.function_count > 0 else 100)
        class_docstring_coverage = (visitor.classes_with_docstrings / visitor.class_count * 100
                                   if visitor.class_count > 0 else 100)
        
        # Generate statistical results
        results = {
            'issues': visitor.issues,
            'stats': {
                'function_count': visitor.function_count,
                'class_count': visitor.class_count,
                'import_count': visitor.import_count,
                'docstring_coverage': {
                    'functions': function_docstring_coverage,
                    'classes': class_docstring_coverage
                },
                'avg_line_length': sum(visitor.line_lengths) / len(visitor.line_lengths) if visitor.line_lengths else 0,
                'max_line_length': max(visitor.line_lengths) if visitor.line_lengths else 0,
                'avg_complexity': sum(visitor.complexity_scores.values()) / len(visitor.complexity_scores) 
                                 if visitor.complexity_scores else 0
            }
        }
        
        return results
    except Exception as e:
        return {'issues': [f"Error analyzing Python style: {str(e)}"], 'stats': {}}


def generate_style_suggestions(analysis_results):
    """
    Generate human-readable suggestions based on style analysis.
    
    Args:
        analysis_results (dict): Results from analyze_python_style
    
    Returns:
        list: Suggestions for improving code style
    """
    suggestions = []
    
    # Process specific issues
    if analysis_results['issues']:
        for issue in analysis_results['issues']:
            suggestions.append(issue)
    
    # Generate statistical suggestions
    stats = analysis_results['stats']
    
    # Line length suggestions
    if stats.get('max_line_length', 0) > 79:
        suggestions.append(f"Some lines exceed PEP 8 recommended length of 79 characters (max: {stats['max_line_length']})")
    
    # Docstring coverage suggestions
    func_coverage = stats.get('docstring_coverage', {}).get('functions', 100)
    class_coverage = stats.get('docstring_coverage', {}).get('classes', 100)
    
    if func_coverage < 80 and stats.get('function_count', 0) > 0:
        suggestions.append(f"Only {func_coverage:.1f}% of functions have docstrings. Consider adding more.")
    
    if class_coverage < 80 and stats.get('class_count', 0) > 0:
        suggestions.append(f"Only {class_coverage:.1f}% of classes have docstrings. Consider adding more.")
    
    # Complexity suggestions
    if stats.get('avg_complexity', 0) > 7:
        suggestions.append(f"Average function complexity ({stats['avg_complexity']:.1f}) is high. Consider simplifying complex functions.")
    
    # If no issues found
    if not suggestions:
        suggestions.append("Code follows Python style conventions well. No style issues detected.")
    
    return suggestions

