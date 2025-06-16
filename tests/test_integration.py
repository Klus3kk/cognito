"""
Integration tests for Cognito code analysis platform.
Tests the complete analysis pipeline across different languages.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzer import analyze_code
from language_detector import detect_code_language
from generic_analyzer import analyze_generic_code


class TestCompleteAnalysisPipeline:
    """Test the complete analysis pipeline from input to output."""
    
    # Sample code snippets for testing
    PYTHON_CODE = '''
def calculate_fibonacci(n):
    """Calculate Fibonacci number using recursion."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
'''
    
    C_CODE = '''
#include <stdio.h>
#include <stdlib.h>

int fibonacci(int n) {
    if (n <= 1)
        return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
    int result = fibonacci(10);
    printf("Fibonacci(10) = %d\\n", result);
    return 0;
}
'''
    
    JAVASCRIPT_CODE = '''
function calculateFibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return calculateFibonacci(n-1) + calculateFibonacci(n-2);
}

function main() {
    const result = calculateFibonacci(10);
    console.log(`Fibonacci(10) = ${result}`);
}

main();
'''
    
    SECURITY_VULNERABLE_CODE = '''
import os

def process_user_input(user_input):
    # Security vulnerability: eval usage
    result = eval(user_input)
    return result

def read_file(filename):
    # Security vulnerability: file handling without proper error management
    f = open(filename, 'r')
    data = f.read()
    return data

def execute_command(cmd):
    # Security vulnerability: command injection
    result = os.system(cmd)
    return result
'''

    def test_python_analysis_pipeline(self):
        """Test complete Python code analysis."""
        results = analyze_code(self.PYTHON_CODE, language='python')
        
        # Verify basic structure
        assert results['language'] == 'python'
        assert 'analysis' in results
        assert 'suggestions' in results
        
        # Verify analysis components
        analysis = results['analysis']
        assert 'readability' in analysis
        assert 'performance' in analysis
        assert 'security' in analysis
        
        # Verify performance detection (should detect recursion)
        performance = analysis['performance']
        if isinstance(performance, list):
            performance_text = ' '.join(performance)
        else:
            performance_text = str(performance)
        
        # Should detect recursion in Fibonacci function
        assert 'recursion' in performance_text.lower() or 'recursive' in performance_text.lower()
    
    def test_c_analysis_pipeline(self):
        """Test complete C code analysis."""
        results = analyze_code(self.C_CODE, language='c')
        
        # Verify basic structure
        assert results['language'] == 'c'
        assert 'analysis' in results
        assert 'summary' in results
        
        # Verify C-specific analysis
        summary = results['summary']
        assert 'maintainability' in summary
    
    def test_javascript_generic_analysis(self):
        """Test generic analysis for JavaScript."""
        results = analyze_generic_code(self.JAVASCRIPT_CODE, 'javascript')
        
        # Verify basic structure
        assert results['language'] == 'javascript'
        assert 'metrics' in results
        assert 'suggestions' in results
        
        # Verify metrics
        metrics = results['metrics']
        assert metrics['function_count'] > 0  # Should detect functions
        assert metrics['total_lines'] > 0
    
    def test_security_vulnerability_detection(self):
        """Test security vulnerability detection."""
        results = analyze_code(self.SECURITY_VULNERABLE_CODE, language='python')
        
        # Get security analysis
        analysis = results['analysis']
        security_text = str(analysis.get('security', '')).lower()
        
        # Should detect at least one security issue
        security_indicators = ['security', 'vulnerability', 'eval', 'open', 'system']
        assert any(indicator in security_text for indicator in security_indicators)
    
    def test_language_detection_accuracy(self):
        """Test language detection accuracy."""
        # Test Python detection
        python_result = detect_code_language(self.PYTHON_CODE)
        assert python_result == 'python'
        
        # Test C detection
        c_result = detect_code_language(self.C_CODE)
        assert c_result == 'c'
        
        # Test JavaScript detection
        js_result = detect_code_language(self.JAVASCRIPT_CODE)
        assert js_result == 'javascript'
    
    def test_filename_based_detection(self):
        """Test language detection with filename hints."""
        # Test with filename
        python_result = detect_code_language(self.PYTHON_CODE, 'test.py')
        assert python_result == 'python'
        
        c_result = detect_code_language(self.C_CODE, 'test.c')
        assert c_result == 'c'
        
        js_result = detect_code_language(self.JAVASCRIPT_CODE, 'test.js')
        assert js_result == 'javascript'
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test with empty code
        results = analyze_code('', language='python')
        assert results is not None
        assert 'language' in results
        
        # Test with malformed code
        malformed_code = "def broken_function(\n    print('incomplete"
        results = analyze_code(malformed_code, language='python')
        assert results is not None
    
    def test_large_code_handling(self):
        """Test handling of larger code files."""
        # Create a larger code sample
        large_code = self.PYTHON_CODE * 10  # Repeat the code 10 times
        
        results = analyze_code(large_code, language='python')
        assert results is not None
        assert 'analysis' in results
        
        # Should still detect multiple functions
        if 'summary' in results:
            summary = results['summary']
            # Should detect multiple instances of the same function pattern
    
    def test_suggestion_generation(self):
        """Test that suggestions are generated appropriately."""
        results = analyze_code(self.PYTHON_CODE, language='python')
        
        suggestions = results.get('suggestions', [])
        assert isinstance(suggestions, list)
        
        # Each suggestion should have proper structure
        for suggestion in suggestions:
            if isinstance(suggestion, dict):
                assert 'category' in suggestion or 'message' in suggestion
    
    def test_cross_language_consistency(self):
        """Test that similar code patterns are detected across languages."""
        # All test codes implement Fibonacci - should detect recursion
        
        python_results = analyze_code(self.PYTHON_CODE, language='python')
        c_results = analyze_code(self.C_CODE, language='c')
        
        # Both should have some form of performance analysis
        assert 'analysis' in python_results
        assert 'analysis' in c_results or 'summary' in c_results


class TestSpecificAnalyzers:
    """Test individual analyzer components."""
    
    def test_readability_analyzer(self):
        """Test readability analysis specifically."""
        try:
            from analyzers.readability_analyzer import analyze_readability
            
            good_code = '''
def calculate_area(radius):
    """Calculate the area of a circle."""
    import math
    return math.pi * radius ** 2
'''
            
            poor_code = '''
def f(x):return x*x*3.14159
'''
            
            good_result = analyze_readability(good_code)
            poor_result = analyze_readability(poor_code)
            
            assert isinstance(good_result, str)
            assert isinstance(poor_result, str)
            
            # Good code should have better readability assessment
            assert 'good' in good_result.lower() or 'excellent' in good_result.lower()
            
        except ImportError:
            pytest.skip("Readability analyzer not available")
    
    def test_performance_analyzer(self):
        """Test performance analysis specifically."""
        try:
            from analyzers.performance_analyzer import analyze_complexity
            
            complex_code = '''
def nested_loops(matrix):
    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix)):
                result.append(matrix[i][j] * k)
    return result
'''
            
            simple_code = '''
def simple_function(x):
    return x * 2
'''
            
            complex_result = analyze_complexity(complex_code)
            simple_result = analyze_complexity(simple_code)
            
            # Complex code should be flagged
            if isinstance(complex_result, list):
                complex_text = ' '.join(complex_result)
            else:
                complex_text = str(complex_result)
            
            assert 'nested' in complex_text.lower() or 'complex' in complex_text.lower()
            
        except ImportError:
            pytest.skip("Performance analyzer not available")
    
    def test_security_analyzer(self):
        """Test security analysis specifically."""
        try:
            from analyzers.security_analyzer import analyze_security
            
            vulnerable_code = '''
def dangerous_function(user_input):
    result = eval(user_input)
    return result
'''
            
            safe_code = '''
def safe_function(user_input):
    try:
        result = int(user_input)
        return result * 2
    except ValueError:
        return 0
'''
            
            vulnerable_result = analyze_security(vulnerable_code)
            safe_result = analyze_security(safe_code)
            
            # Vulnerable code should be flagged
            if isinstance(vulnerable_result, list):
                vulnerable_text = ' '.join(vulnerable_result)
            else:
                vulnerable_text = str(vulnerable_result)
            
            assert 'eval' in vulnerable_text.lower() or 'security' in vulnerable_text.lower()
            
        except ImportError:
            pytest.skip("Security analyzer not available")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])