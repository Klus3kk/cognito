from analyzers.performance_analyzer import analyze_complexity

def test_simple_loop():
    code = "for i in range(10): print(i)"
    result = analyze_complexity(code)
    assert "Code complexity looks good." in result

def test_nested_loops():
    code = "for i in range(10):\n    for j in range(10):\n        for k in range(10):\n            print(i, j, k)"
    result = analyze_complexity(code)
    assert "Code contains deeply nested loops." in result

def test_recursion():
    code = "def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)"
    result = analyze_complexity(code)
    assert "Recursive function detected." in result
