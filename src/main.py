"""
Enhanced main module for Cognito with code correction functionality, LLM integration,
and a feedback-driven learning system.

This is an improved version of the main.py that adds code correction features,
LLM capabilities, and continuous improvement through user feedback.
"""

import os
import sys
import time
import logging
import argparse
from colorama import init, Fore, Style
from pathlib import Path
from datetime import datetime

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
    
    # Try to import feedback collector - this is optional
    try:
        from feedback.collector import FeedbackCollector
        feedback_collector_available = True
    except ImportError:
        feedback_collector_available = False
        
    # Try to import metrics reporter - this is optional
    try:
        from reports.improvement_metrics import ImprovementMetricsReporter
        metrics_reporter_available = True
    except ImportError:
        metrics_reporter_available = False
        
    # Try to import learning LLM enhancer - this is optional
    try:
        from llm.learning_enhancer import LearningLLMIntegration
        learning_llm_available = True
    except ImportError:
        learning_llm_available = False
        
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
    {Fore.GREEN}AI-Powered Code Review Assistant {Style.RESET_ALL}v0.3.0
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
    """Format a suggestion with appropriate color based on category and content."""
    if isinstance(suggestion, list):
        result = ""
        for item in suggestion:
            if is_positive_message(item):
                result += f"{Fore.GREEN}✓ {item}{Style.RESET_ALL}\n"
            else:
                result += f"{Fore.RED}✗ {item}{Style.RESET_ALL}\n"
        return result.strip()
    else:
        if is_positive_message(suggestion):
            return f"{Fore.GREEN}✓ {suggestion}{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}✗ {suggestion}{Style.RESET_ALL}"

def is_positive_message(message):
    """
    Determine if a message represents a positive/good result.
    
    Args:
        message (str): The message to check
        
    Returns:
        bool: True if message is positive, False if it indicates an issue
    """
    if not isinstance(message, str):
        return False
    
    message_lower = message.lower()
    
    # Positive indicators - these should show as green ✓
    positive_patterns = [
        "looks good",
        "no issues detected", 
        "no problems found",
        "passes",
        "security analysis: code passes",
        "owasp security checks",
        "code complexity looks good",
        "no style issues detected",
        "no common anti-patterns detected",
        "no common security issues detected",
        "code readability looks good"
    ]
    
    # Check if message contains positive indicators
    for pattern in positive_patterns:
        if pattern in message_lower:
            return True
    
    # Special case for security messages that mention "passes"
    if "security analysis:" in message_lower and "passes" in message_lower:
        return True
    
    return False

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

def process_suggestion_feedback(suggestion, all_feedback):
    """Get user feedback on a suggestion and update the learning system."""
    if not feedback_collector_available:
        return all_feedback
        
    print(f"\n{Fore.CYAN}Was this suggestion helpful? (y/n): {Style.RESET_ALL}")
    feedback = input().lower()
    
    if feedback == 'y':
        accepted = True
        print(f"{Fore.GREEN}Great! Any comments on why this was helpful? (Enter to skip){Style.RESET_ALL}")
    else:
        accepted = False
        print(f"{Fore.YELLOW}Thanks for the feedback. Any comments on why this wasn't helpful? (Enter to skip){Style.RESET_ALL}")
    
    comment = input()
    
    # Add to the feedback system
    try:
        feedback_collector = FeedbackCollector()
        feedback_collector.add_feedback(suggestion, accepted, comment if comment else None)
        
        # Show improvement metrics
        metrics = feedback_collector.get_metrics()
        print(f"\n{Fore.CYAN}System learning progress:{Style.RESET_ALL}")
        print(f"Total suggestions processed: {metrics['total_suggestions']}")
        print(f"Current acceptance rate: {metrics['acceptance_rate']}%")
        
        if metrics['total_suggestions'] > 10:
            improvement = feedback_collector.get_improvement_metrics(interval_days=30)
            if improvement['previous_period']['total'] > 0:
                print(f"Suggestion quality improved by {improvement['acceptance_improvement_percentage']}% over the last month")
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        print(f"{Fore.YELLOW}Feedback recorded, but could not update learning system: {str(e)}{Style.RESET_ALL}")
    
    all_feedback += f"User feedback on suggestion: {'Accepted' if accepted else 'Rejected'}\n"
    if comment:
        all_feedback += f"Comment: {comment}\n\n"
    
    return all_feedback

def generate_improvement_report():
    """Generate and display a report of improvement metrics."""
    if not metrics_reporter_available:
        print(f"{Fore.RED}Error: Metrics reporter module not available.{Style.RESET_ALL}")
        return
        
    try:
        reporter = ImprovementMetricsReporter()
        report_file = reporter.generate_improvement_report()
        print(f"\n{Fore.GREEN}Improvement metrics report generated: {report_file}{Style.RESET_ALL}")
        
        # Display CV-worthy metrics
        metrics = FeedbackCollector().get_metrics()
        if metrics["total_suggestions"] > 0:
            print(f"\n{Fore.CYAN}CV-Worthy Metrics:{Style.RESET_ALL}")
            
            # Overall acceptance rate
            print(f"• Overall suggestion acceptance rate: {metrics['acceptance_rate']}%")
            
            # If we have enough data for improvement metrics
            if metrics["total_suggestions"] > 10:
                try:
                    monthly = reporter.feedback_collector.get_improvement_metrics(interval_days=30)
                    if monthly["previous_period"]["total"] > 0:
                        improvement = monthly["acceptance_improvement_percentage"]
                        print(f"• Improved suggestion acceptance rate by {improvement}% over one month")
                except:
                    pass
                    
            # Show category-specific metrics
            for category in ["Security", "Readability", "Performance"]:
                try:
                    cat_metrics = reporter.feedback_collector.get_suggestion_performance(category)
                    if cat_metrics["total"] > 5:
                        print(f"• {category} suggestion acceptance rate: {cat_metrics['acceptance_rate']}%")
                except:
                    pass
        else:
            print(f"\n{Fore.YELLOW}Not enough feedback data yet to generate metrics.{Style.RESET_ALL}")
            print("Use Cognito more and provide feedback on suggestions to build up metrics.")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        print(f"{Fore.RED}Error generating report: {str(e)}{Style.RESET_ALL}")

def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cognito - AI-powered code review assistant")
    parser.add_argument("--file", help="Path to code file to analyze")
    parser.add_argument("--language", help="Force language detection (python, c)")
    parser.add_argument("--output", help="Output file for analysis results")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to enhance analysis")
    parser.add_argument("--report", action="store_true", help="Generate improvement metrics report")
    parser.add_argument("--adaptive", action="store_true", help="Use feedback-adaptive LLM mode")
    args = parser.parse_args()
    
    # Handle report generation request
    if args.report:
        generate_improvement_report()
        return
    
    try:
        # If direct file input is provided, analyze it
        if args.file:
            try:
                with open(args.file, 'r') as file:
                    code_input = file.read()
                    filename = os.path.basename(args.file)
                    
                    # Detect or force language
                    language = args.language or detect_language(code_input)
                    
                    # Select the appropriate LLM enhancer based on args
                    if args.use_llm and args.adaptive and learning_llm_available:
                        print(f"Analyzing {filename} as {language} with adaptive LLM enhancement...")
                        # Use the learning LLM integration for enhanced analysis
                        from llm.learning_enhancer import LearningLLMIntegration
                        learning_enhancer = LearningLLMIntegration()
                        analysis_results = analyze_code(code_input, filename, language, use_llm=True, 
                                                       llm_integration=learning_enhancer)
                    else:
                        # Use standard analysis
                        print(f"Analyzing {filename} as {language}{' with LLM' if args.use_llm else ''}...")
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
                        
                        # Get feedback if available
                        if feedback_collector_available:
                            process_suggestion_feedback(suggestion, "")
                    
                    # If LLM was used, print those insights
                    if args.use_llm and 'code_explanation' in analysis_results:
                        print("\nCode Explanation:")
                        print(analysis_results['code_explanation'])
                        
                        if 'ai_review' in analysis_results:
                            print("\nAI Review:")
                            print(analysis_results['ai_review'])
                        
                        # If adaptive mode was used, print adaptation insights
                        if args.adaptive and 'adaptation_insights' in analysis_results:
                            insights = analysis_results['adaptation_insights']
                            if insights.get('total_suggestions_processed', 0) > 0:
                                print("\nAdaptation Insights:")
                                print(f"System has processed {insights.get('total_suggestions_processed', 0)} suggestions to date")
                                improvement = insights.get('monthly_improvement', {}).get('acceptance_improvement_percentage')
                                if improvement:
                                    print(f"Suggestion quality improved by {improvement}% over the last month")
                        
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
            print("3. View improvement metrics")
            print("4. Exit")
            
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
                generate_improvement_report()
                continue
                
            elif choice == '4':
                print(f"\n{Fore.GREEN}Thank you for using Cognito! Goodbye!{Style.RESET_ALL}")
                sys.exit(0)
            
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                continue
            
            # Detect language
            language = detect_language(code_input)
            print(f"\n{Fore.CYAN}Detected language: {language}{Style.RESET_ALL}")
            
            # Check if the user wants to use LLM enhancement
            use_llm = False
            if not args.use_llm:  # Only ask if not already specified in args
                llm_option = input(f"{Fore.CYAN}Would you like to use AI enhancement? (y/n): {Style.RESET_ALL}").lower()
                use_llm = llm_option == 'y'
            else:
                use_llm = args.use_llm
                
            # Check if the user wants to use adaptive mode
            use_adaptive = False
            if use_llm and learning_llm_available and not args.adaptive:  # Only ask if using LLM and not in args
                adaptive_option = input(f"{Fore.CYAN}Use adaptive AI that learns from feedback? (y/n): {Style.RESET_ALL}").lower()
                use_adaptive = adaptive_option == 'y'
            else:
                use_adaptive = args.adaptive
            
            print(f"{Fore.CYAN}Analyzing code with ML-powered assistant...{Style.RESET_ALL}")
            analysis_start_time = time.time()

            # Analyze code
            print(f"{Fore.CYAN}Analyzing code{' with adaptive AI' if use_adaptive else ' with LLM' if use_llm else ''}...{Style.RESET_ALL}")
            
            # Analysis process
            all_feedback = ""
            feedback_items = {'code': code_input}
            
            # Use appropriate analyzer based on options
            if use_llm and use_adaptive and learning_llm_available:
                from llm.learning_enhancer import LearningLLMIntegration
                learning_enhancer = LearningLLMIntegration()
                analysis_results = analyze_code(code_input, filename, language, use_llm=True,
                                               llm_integration=learning_enhancer)
            else:
                analysis_results = analyze_code(code_input, filename, language, use_llm=use_llm)
            
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
            if use_llm:
                print_section_header("AI-Enhanced Insights")
                if 'code_explanation' in analysis_results:
                    print(f"{Fore.GREEN}✓ Code Explanation:{Style.RESET_ALL}")
                    print(analysis_results['code_explanation'])
                    all_feedback += f"AI Code Explanation:\n{analysis_results['code_explanation']}\n\n"
                
                if 'ai_review' in analysis_results:
                    print(f"{Fore.GREEN}✓ AI Review:{Style.RESET_ALL}")
                    print(analysis_results['ai_review'])
                    all_feedback += f"AI Review:\n{analysis_results['ai_review']}\n\n"
                    
                # If adaptive mode was used, show adaptation insights
                if use_adaptive and 'adaptation_insights' in analysis_results:
                    insights = analysis_results['adaptation_insights']
                    if insights.get('total_suggestions_processed', 0) > 0:
                        print(f"\n{Fore.GREEN}✓ Adaptation Insights:{Style.RESET_ALL}")
                        print(f"System has processed {insights.get('total_suggestions_processed', 0)} suggestions")
                        
                        improvement = insights.get('monthly_improvement', {}).get('acceptance_improvement_percentage')
                        if improvement:
                            print(f"Suggestion quality improved by {improvement}% over the last month")
                            all_feedback += f"Suggestion quality improved by {improvement}% over the last month\n\n"
            
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
                    
                    # Get feedback on this correction
                    if feedback_collector_available:
                        print(f"\n{Fore.CYAN}Was this correction helpful? (y/n): {Style.RESET_ALL}")
                        correction_feedback = input().lower()
                        correction_suggestion = {
                            "category": "Code Correction",
                            "message": "Automatic code correction",
                            "priority": "medium"
                        }
                        all_feedback = process_suggestion_feedback(correction_suggestion, all_feedback)
                    
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
                
            analysis_end_time = time.time()
            analysis_duration = analysis_end_time - analysis_start_time
            
            print(f"\n{Fore.GREEN}⚡ Analysis completed in {analysis_duration:.1f} seconds{Style.RESET_ALL}")

            # Add speed comparison
            if analysis_duration < 30:
                manual_time_estimate = 180  # 3 minutes for manual review
                speed_improvement = ((manual_time_estimate - analysis_duration) / manual_time_estimate) * 100
                print(f"{Fore.YELLOW}🚀 {speed_improvement:.0f}% faster than manual code review{Style.RESET_ALL}")

            # Process suggestion feedback
            if feedback_collector_available and analysis_results.get('suggestions'):
                print_section_header("Suggestion Feedback")
                print(f"{Fore.CYAN}Please provide feedback on our suggestions to help improve the system:{Style.RESET_ALL}")
                
                # Get feedback on up to 3 suggestions
                for i, suggestion in enumerate(analysis_results.get('suggestions', [])[:3]):
                    category = suggestion.get('category', '')
                    message = suggestion.get('message', '')
                    print(f"\n{Fore.YELLOW}Suggestion {i+1}: [{category}] {message}{Style.RESET_ALL}")
                    all_feedback = process_suggestion_feedback(suggestion, all_feedback)
            
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