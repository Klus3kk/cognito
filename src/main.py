from analyzers.readability_analyzer import analyze_readability
from analyzers.performance_analyzer import analyze_complexity, analyze_memory_usage

def main():
    print("Welcome to Cognito: AI-Powered Code Review Assistant - Master Edition!")
    print("Supported languages: Python, C")
    print("Please enter your code below (type 'exit' to quit):")

    while True:
        code_input = input(">>> ")
        if code_input.lower() == "exit":
            print("Exiting Cognito. Thank you for using our code review assistant!")
            break

        # Readability Analysis
        readability_feedback = analyze_readability(code_input)
        print("Readability Analysis:", readability_feedback)

        # Performance Analysis (for now Big-O Complexity)
        complexity_feedback = analyze_complexity(code_input)
        print("Performance Analysis:", complexity_feedback)

        # Memory Usage Analysis
        memory_feedback = analyze_memory_usage(code_input)
        print("Memory Usage Analysis: ", memory_feedback)