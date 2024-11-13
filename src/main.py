def main():
    print("Welcome to Cognito: AI-Powered Code Review Assistant")
    print("Supported languages: Python, C")
    print("Please enter your code below (type 'exit' to quit):")

    while True:
        code_input = input(">>> ")
        if code_input.lower() == "exit":
            print("Exiting Cognito. Thank you for using our code review assistant!")
            break
        
        # Placeholder for analysis logic
        print("Analyzing your code...")
        # Here it would call the analysis functions for Python or C
        # For now, I just echo the code
        print("Code received:", code_input)
        print("Suggestions will be displayed here... (soon [maybe not that soon])")

if __name__ == "__main__":
    main()
