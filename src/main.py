from models.nlp_model import analyze_readability

def main():
    print("Welcome to Cognito: AI-Powered Code Review Assistant - Master Edition!")
    print("Supported languages: Python, C")
    print("Please enter your code below (type 'exit' to quit):")

    while True:
        code_input = input(">>> ")
        if code_input.lower() == "exit":
            print("Exiting Cognito. Thank you for using our code review assistant!")
            break

        # Analyze readability with NLP model
        readability_feedback = analyze_readability(code_input)
        print("Readability Analysis:", readability_feedback)
