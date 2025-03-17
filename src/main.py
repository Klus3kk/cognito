"""
Enhanced main module for Cognito with code correction functionality and LLM integration.

This is an improved version of the main.py that adds code correction features and LLM capabilities.
"""

import os
import sys
import logging
import argparse
from colorama import init, Fore, Style
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize colorama for cross-platform colored terminal output
init()

# Add the current directory to path to ensure imports work
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import analyzers
try:
    from analyzers.readability_analyzer import analyze_readability
    from analyzers.performance_analyzer import analyze_complexity, analyze_memory_usage
    from analyzers.security_analyzer import analyze_security, generate_security_suggestion
    from language_detector import detect_code_language
    from analyzer import analyze_code
    
    # Import the code corrector
    from code_correction import CodeCorrector
    from code_correction import extract_issues_from_feedback
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    print(f"{Fore.RED}Error: Could not import required modules. Make sure you're running from the project root.{Style.RESET_ALL}")
    print(f"{Fore.RED}Missing: {e}{Style.RESET_ALL}")
    sys.exit(1)

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
    {Fore.GREEN}AI-Powered Code Review Assistant {Style.RESET_ALL}v0.2.0
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
        return detect_code_language(code)
    except Exception as e:
        logger.warning(f"Language detector error: {e}")
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

def save_feedback(code, feedback, corrected_code=None, filename=None):
    """Save the code, feedback, and corrected code to a file."""
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
        
        if corrected_code:
            file.write("\n\n=== CORRECTED CODE ===\n\n")
            file.write(corrected_code)
    
    print(f"\n{Fore.GREEN}Feedback saved to {filename}{Style.RESET_ALL}")

def highlight_code_with_issues(code, issues):
    """Highlight lines with issues in the code."""
    lines = code.split('\n')
    result = []
    
    # Track which lines have issues
    issue_lines = {}
    for issue in issues:
        if hasattr(issue, 'get'):
            line = issue.get('line', 0)
            if line > 0 and line <= len(lines):
                if line not in issue_lines:
                    issue_lines[line] = []
                issue_lines[line].append(issue)
    
    # Add each line, highlighting those with issues
    for i, line in enumerate(lines, 1):
        if i in issue_lines:
            result.append(f"{Fore.RED}{i:4d} | {line}{Style.RESET_ALL}")
            # Add issue description
            for issue in issue_lines[i]:
                if hasattr(issue, 'get'):
                    message = issue.get('message', '')
                    if message:
                        result.append(f"     | {Fore.RED}^ {message}{Style.RESET_ALL}")
        else:
            result.append(f"{i:4d} | {line}")
    
    return '\n'.join(result)

def find_line_with_pattern(code, pattern):
    """Find the line number containing a regex pattern."""
    import re
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if re.search(pattern, line):
            return i
    
    # If we're looking for a pattern that might span multiple lines
    code_flat = '\n'.join(lines)
    match = re.search(pattern, code_flat)
    if match:
        # Count lines up to the match position
        return code_flat[:match.start()].count('\n') + 1
    
    return 0

def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cognito - AI-powered code review assistant")
    parser.add_argument("--file", help="Path to code file to analyze")
    parser.add_argument("--language", help="Force language detection (python, c)")
    parser.add_argument("--output", help="Output file for analysis results")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to enhance analysis")
    args = parser.parse_args()
    
    try:
        # If direct file input is provided, analyze it
        if args.file:
            try:
                with open(args.file, 'r') as file:
                    code_input = file.read()
                    filename = os.path.basename(args.file)
                    
                    # Detect or force language
                    language = args.language or detect_language(code_input)
                    
                    # Analyze code
                    print(f"Analyzing {filename} as {language}...")
                    analysis_results = analyze_code(code_input, filename, language, use_llm=args.use_llm)
                    
                    # Print summary
                    print("\nAnalysis Summary:")
                    for key, value in analysis_results.get('summary', {}).items():
                        print(f"- {key}: {value}")
                    
                    # Print suggestions
                    print("\nSuggestions:")
                    for suggestion in analysis_results.get('suggestions', []):
                        category = suggestion.get('category', '')
                        message = suggestion.get('message', '')
                        priority = suggestion.get('priority', '')
                        print(f"[{priority.upper()}] {category}: {message}")
                    
                    # If LLM was used, print those insights
                    if args.use_llm and 'code_explanation' in analysis_results:
                        print("\nCode Explanation:")
                        print(analysis_results['code_explanation'])
                        
                    # Save results if output file is specified
                    if args.output:
                        save_feedback(code_input, str(analysis_results), None, args.output)
                        
                    return
            except Exception as e:
                print(f"Error analyzing file: {str(e)}")
                return
        
        # Interactive mode
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
            print(f"{Fore.CYAN}Analyzing code{' with LLM' if args.use_llm else ''}...{Style.RESET_ALL}")
            
            # Analysis process
            all_feedback = ""
            feedback_items = {'code': code_input}
            
            # Use unified analyzer
            analysis_results = analyze_code(code_input, filename, language, use_llm=args.use_llm)
            
            # Readability Analysis
            print_section_header("Readability Analysis")
            try:
                readability_feedback = analysis_results['analysis'].get('readability', 'No readability analysis available.')
                formatted_feedback = format_suggestion(readability_feedback, "readability")
                print(formatted_feedback)
                all_feedback += f"Readability Analysis:\n{readability_feedback}\n\n"
                feedback_items['readability'] = readability_feedback
            except Exception as e:
                error_msg = f"Error in readability analysis: {str(e)}"
                print(f"{Fore.RED}✗ {error_msg}{Style.RESET_ALL}")
                all_feedback += f"Readability Analysis:\n{error_msg}\n\n"
                logger.error(error_msg, exc_info=True)
            
            # Performance Analysis
            print_section_header("Performance Analysis")
            try:
                performance_feedback = analysis_results['analysis'].get('performance', 'No performance analysis available.')
                formatted_feedback = format_suggestion(performance_feedback, "performance")
                print(formatted_feedback)
                all_feedback += f"Performance Analysis:\n{performance_feedback}\n\n"
                feedback_items['performance'] = performance_feedback
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
                feedback_items['memory'] = memory_feedback
            except Exception as e:
                error_msg = f"Error in memory analysis: {str(e)}"
                print(f"{Fore.RED}✗ {error_msg}{Style.RESET_ALL}")
                all_feedback += f"Memory Usage Analysis:\n{error_msg}\n\n"
                logger.error(error_msg, exc_info=True)
            
            # Security Analysis
            print_section_header("Security Analysis")
            try:
                security_feedback = analysis_results['analysis'].get('security', 'No security analysis available.')
                formatted_feedback = format_suggestion(security_feedback, "security")
                print(formatted_feedback)
                all_feedback += f"Security Analysis:\n{security_feedback}\n\n"
                feedback_items['security'] = security_feedback
            except Exception as e:
                error_msg = f"Error in security analysis: {str(e)}"
                print(f"{Fore.RED}✗ {error_msg}{Style.RESET_ALL}")
                all_feedback += f"Security Analysis:\n{error_msg}\n\n"
                logger.error(error_msg, exc_info=True)
            
            # Display LLM insights if available
            if args.use_llm:
                print_section_header("AI-Enhanced Insights")
                if 'code_explanation' in analysis_results:
                    print(f"{Fore.GREEN}✓ Code Explanation:{Style.RESET_ALL}")
                    print(analysis_results['code_explanation'])
                    all_feedback += f"AI Code Explanation:\n{analysis_results['code_explanation']}\n\n"
                
                if 'ai_review' in analysis_results:
                    print(f"{Fore.GREEN}✓ AI Review:{Style.RESET_ALL}")
                    print(analysis_results['ai_review'])
                    all_feedback += f"AI Review:\n{analysis_results['ai_review']}\n\n"
            
            # Extract issues for code correction
            issues = extract_issues_from_feedback(feedback_items)
            
            # Code Correction
            print_section_header("Code Correction")
            try:
                corrector = CodeCorrector(language)
                corrected_code = corrector.correct_code(code_input, issues)
                
                # Check if code was actually corrected
                if corrected_code != code_input:
                    print(f"{Fore.GREEN}✓ Code can be improved! Here's the highlighted version with issues:{Style.RESET_ALL}\n")
                    highlighted_code = highlight_code_with_issues(code_input, issues)
                    print(highlighted_code)
                    
                    print(f"\n{Fore.GREEN}✓ Corrected code:{Style.RESET_ALL}\n")
                    print(corrected_code)
                    
                    # Show the differences
                    print(f"\n{Fore.CYAN}Would you like to see a diff of the changes? (y/n): {Style.RESET_ALL}")
                    show_diff = input().lower()
                    if show_diff == 'y':
                        diff = corrector.generate_diff(code_input, corrected_code)
                        print(f"\n{diff}")
                else:
                    print(f"{Fore.YELLOW}✓ No automatic corrections available for the identified issues.{Style.RESET_ALL}")
                    corrected_code = None
            except Exception as e:
                error_msg = f"Error in code correction: {str(e)}"
                print(f"{Fore.RED}✗ {error_msg}{Style.RESET_ALL}")
                logger.error(error_msg, exc_info=True)
                corrected_code = None
            
            # Ask if user wants to save feedback
            print_separator()
            save_option = input(f"\n{Fore.CYAN}Would you like to save this feedback? (y/n): {Style.RESET_ALL}")
            if save_option.lower() == 'y':
                custom_filename = input(f"{Fore.CYAN}Enter filename (or press Enter for default): {Style.RESET_ALL}")
                if custom_filename:
                    save_feedback(code_input, all_feedback, corrected_code, custom_filename)
                else:
                    default_name = f"cognito_feedback_{filename.split('.')[0]}.txt" if filename else "cognito_feedback.txt"
                    save_feedback(code_input, all_feedback, corrected_code, default_name)

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