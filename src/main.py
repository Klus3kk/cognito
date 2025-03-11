import os
import sys
import logging
from colorama import init, Fore, Style
from analyzers.readability_analyzer import analyze_readability
from analyzers.performance_analyzer import analyze_complexity, analyze_memory_usage
from analyzers.security_analyzer import analyze_security, generate_security_suggestion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize colorama for cross-platform colored terminal output
init()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_logo():
    """Display the ASCII art logo for Cognito."""
    logo = f"""
    {Fore.CYAN}
     ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗████████╗ ██████╗ 
    ██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║╚══██╔══╝██╔═══██╗
    ██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║   ██║   ██║
    ██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║   ██║   ██║
    ╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║   ██║   ╚██████╔╝
     ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝    ╚═════╝ 
    {Style.RESET_ALL}
    {Fore.GREEN}AI-Powered Code Review Assistant {Style.RESET_ALL}v0.1.0
    """
    print(logo)

def print_separator():
    """Print a separator line."""
    print(f"{Fore.BLUE}{'='*70}{Style.RESET_ALL}")

def print_section_header(title):
    """Print a section header with formatting."""
    print(f"\n{Fore.YELLOW}▶ {title}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'-'*50}{Style.RESET_ALL}")

def format_suggestion(suggestion, category):
    """Format a suggestion with appropriate color based on category."""
    if isinstance(suggestion, list):
        result = ""
        for item in suggestion:
            if "good" in item.lower() or "no issues" in item.lower():
                result += f"{Fore.GREEN}✓ {item}{Style.RESET_ALL}\n"
            else:
                result += f"{Fore.RED}✗ {item}{Style.RESET_ALL}\n"
        return result.strip()
    else:
        if "good" in suggestion.lower() or "no issues" in suggestion.lower():
            return f"{Fore.GREEN}✓ {suggestion}{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}✗ {suggestion}{Style.RESET_ALL}"

def detect_language(code):
    """Attempt to detect the programming language of the code."""
    try:
        # Import the language detector here to avoid circular imports
        from language_detector import detect_code_language
        return detect_code_language(code)
    except ImportError:
        logger.warning("Language detector not available, using simple detection")
        # Fallback to simple detection
        if '#include' in code and ('{' in code and '}' in code and ';' in code):
            return 'C'
        elif 'def ' in code or 'class ' in code or 'import ' in code:
            return 'Python'
        else:
            return 'Unknown'

def handle_file_input():
    """Handle file input option."""
    filepath = input(f"{Fore.CYAN}Enter file path: {Style.RESET_ALL}")
    try:
        with open(filepath, 'r') as file:
            return file.read(), os.path.basename(filepath)
    except Exception as e:
        print(f"{Fore.RED}Error reading file: {str(e)}{Style.RESET_ALL}")
        return None, None

def save_feedback(code, feedback, filename=None):
    """Save the code and feedback to a file."""
    if not filename:
        filename = "cognito_feedback.txt"
    
    with open(filename, 'w') as file:
        file.write("=== CODE ANALYZED ===\n\n")
        file.write(code)
        file.write("\n\n=== FEEDBACK ===\n\n")
        # Remove ANSI color codes for file output
        clean_feedback = feedback.replace(Fore.GREEN, '').replace(Fore.RED, '')
        clean_feedback = clean_feedback.replace(Fore.YELLOW, '').replace(Fore.BLUE, '')
        clean_feedback = clean_feedback.replace(Style.RESET_ALL, '')
        file.write(clean_feedback)
    
    print(f"\n{Fore.GREEN}Feedback saved to {filename}{Style.RESET_ALL}")

def main():
    """Main application entry point."""
    try:
        clear_screen()
        print_logo()
        print_separator()
        
        print(f"{Fore.CYAN}Welcome to Cognito! Let's improve your code together.{Style.RESET_ALL}")
        print(f"Currently supporting: {Fore.GREEN}Python{Style.RESET_ALL}, {Fore.GREEN}C{Style.RESET_ALL}")
        
        while True:
            print_separator()
            print(f"\n{Fore.CYAN}Choose an option:{Style.RESET_ALL}")
            print("1. Enter code snippet")
            print("2. Load code from file")
            print("3. Exit")
            
            choice = input(f"\n{Fore.CYAN}> {Style.RESET_ALL}")
            
            if choice == '1':
                print(f"\n{Fore.CYAN}Enter your code below (type 'DONE' on a new line when finished):{Style.RESET_ALL}\n")
                code_lines = []
                while True:
                    line = input()
                    if line == 'DONE':
                        break
                    code_lines.append(line)
                
                code_input = '\n'.join(code_lines)
                if not code_input.strip():
                    print(f"{Fore.RED}No code entered.{Style.RESET_ALL}")
                    continue
                
                filename = None
            
            elif choice == '2':
                code_input, filename = handle_file_input()
                if not code_input:
                    continue
            
            elif choice == '3':
                print(f"\n{Fore.GREEN}Thank you for using Cognito! Goodbye!{Style.RESET_ALL}")
                sys.exit(0)
            
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                continue
            
            # Detect language
            language = detect_language(code_input)
            print(f"\n{Fore.CYAN}Detected language: {language}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Analyzing code...{Style.RESET_ALL}")
            
            # Analysis process
            all_feedback = ""
            
            # Readability Analysis
            print_section_header("Readability Analysis")
            try:
                readability_feedback = analyze_readability(code_input)
                formatted_feedback = format_suggestion(readability_feedback, "readability")
                print(formatted_feedback)
                all_feedback += f"Readability Analysis:\n{readability_feedback}\n\n"
            except Exception as e:
                error_msg = f"Error in readability analysis: {str(e)}"
                print(f"{Fore.RED}✗ {error_msg}{Style.RESET_ALL}")
                all_feedback += f"Readability Analysis:\n{error_msg}\n\n"
                logger.error(error_msg, exc_info=True)
            
            # Performance Analysis
            print_section_header("Performance Analysis")
            try:
                complexity_feedback = analyze_complexity(code_input)
                formatted_feedback = format_suggestion(complexity_feedback, "performance")
                print(formatted_feedback)
                all_feedback += f"Performance Analysis:\n{complexity_feedback}\n\n"
            except Exception as e:
                error_msg = f"Error in performance analysis: {str(e)}"
                print(f"{Fore.RED}✗ {error_msg}{Style.RESET_ALL}")
                all_feedback += f"Performance Analysis:\n{error_msg}\n\n"
                logger.error(error_msg, exc_info=True)
            
            # Memory Analysis
            print_section_header("Memory Usage Analysis")
            try:
                memory_feedback = analyze_memory_usage(code_input)
                formatted_feedback = format_suggestion(memory_feedback, "memory")
                print(formatted_feedback)
                all_feedback += f"Memory Usage Analysis:\n{memory_feedback}\n\n"
            except Exception as e:
                error_msg = f"Error in memory analysis: {str(e)}"
                print(f"{Fore.RED}✗ {error_msg}{Style.RESET_ALL}")
                all_feedback += f"Memory Usage Analysis:\n{error_msg}\n\n"
                logger.error(error_msg, exc_info=True)
            
            # Security Analysis
            print_section_header("Security Analysis")
            try:
                security_issues = analyze_security(code_input)
                security_feedback = generate_security_suggestion(security_issues)
                formatted_feedback = format_suggestion(security_issues, "security")
                print(formatted_feedback)
                all_feedback += f"Security Analysis:\n{security_feedback}\n\n"
            except Exception as e:
                error_msg = f"Error in security analysis: {str(e)}"
                print(f"{Fore.RED}✗ {error_msg}{Style.RESET_ALL}")
                all_feedback += f"Security Analysis:\n{error_msg}\n\n"
                logger.error(error_msg, exc_info=True)
            
            # Ask if user wants to save feedback
            print_separator()
            save_option = input(f"\n{Fore.CYAN}Would you like to save this feedback? (y/n): {Style.RESET_ALL}")
            if save_option.lower() == 'y':
                custom_filename = input(f"{Fore.CYAN}Enter filename (or press Enter for default): {Style.RESET_ALL}")
                if custom_filename:
                    save_feedback(code_input, all_feedback, custom_filename)
                else:
                    default_name = f"cognito_feedback_{filename.split('.')[0]}.txt" if filename else "cognito_feedback.txt"
                    save_feedback(code_input, all_feedback, default_name)

    except KeyboardInterrupt:
        print(f"\n\n{Fore.GREEN}Exiting Cognito. Thank you for using our code review assistant!{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\n{Fore.RED}An unexpected error occurred: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Check the logs for more details.{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()