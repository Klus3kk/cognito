"""
Main module for Cognito 
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to path to ensure imports work
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import core modules
try:
    from language_detector import LanguageDetector, get_supported_languages, get_language_info
    from generic_analyzer import analyze_generic_code, get_language_support_info
    from clean_styling import CleanStyler, format_message, format_suggestion, print_analysis_results
    
    # Import existing analyzers
    from analyzers.readability_analyzer import analyze_readability
    from analyzers.performance_analyzer import analyze_complexity, analyze_memory_usage
    from analyzers.security_analyzer import analyze_security, generate_security_suggestion
    from analyzer import analyze_code
    
    # Import optional modules
    try:
        from code_correction import CodeCorrector, extract_issues_from_feedback
        code_correction_available = True
    except ImportError:
        code_correction_available = False
        
    try:
        from feedback.collector import FeedbackCollector
        feedback_collector_available = True
    except ImportError:
        feedback_collector_available = False
        
    try:
        from reports.improvement_metrics import ImprovementMetricsReporter
        metrics_reporter_available = True
    except ImportError:
        metrics_reporter_available = False
        
    try:
        from llm.learning_enhancer import LearningLLMIntegration
        learning_llm_available = True
    except ImportError:
        learning_llm_available = False
        
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    print(f"Error: Could not import required modules. Make sure you're running from the project root.")
    print(f"Missing: {e}")
    sys.exit(1)

# Initialize styling
styler = CleanStyler()

def check_recursion_limit():
    """Check and set appropriate recursion limit."""
    current_limit = sys.getrecursionlimit()
    if current_limit < 3000:
        sys.setrecursionlimit(3000)
        print(f"Increased recursion limit from {current_limit} to 3000")

def detect_language(code, filename=None):
    """Enhanced language detection with detailed results."""
    try:
        detector = LanguageDetector()
        result = detector.detect_language(code, filename)
        return result
    except Exception as e:
        logger.warning(f"Language detector error: {e}")
        # Fallback to simple detection
        if '#include' in code and ('{' in code and '}' in code and ';' in code):
            return {'language': 'c', 'confidence': 60.0}
        elif 'def ' in code or 'class ' in code or 'import ' in code:
            return {'language': 'python', 'confidence': 70.0}
        else:
            return {'language': 'unknown', 'confidence': 0.0}

def analyze_code(code, filename=None, language=None, use_llm=False, use_adaptive=False):
    """Enhanced code analysis supporting multiple languages - FIXED RECURSION."""
    
    # Detect language if not specified
    if not language:
        detection_result = detect_language(code, filename)
        language = detection_result['language']
        confidence = detection_result.get('confidence', 0)
        
        print(format_message(f"Detected language: {language.title()} ({confidence:.1f}% confidence)", 'info'))
        
        # Show alternatives if confidence is low
        if confidence < 70 and 'alternatives' in detection_result:
            alternatives = detection_result['alternatives'][:3]
            if alternatives:
                alt_str = ", ".join([alt.title() for alt in alternatives if alt != language])
                print(format_message(f"Alternative possibilities: {alt_str}", 'warning'))
    
    # FIXED: Import analyze_code directly to prevent recursion
    try:
        if language.lower() in ['python', 'c']:
            # Use existing specialized analyzers - IMPORT DIRECTLY
            from analyzer import CodeAnalyzer
            analyzer = CodeAnalyzer()
            return analyzer.analyze(code, filename, language, use_llm)
        else:
            # Use generic analyzer for other languages
            from generic_analyzer import analyze_generic_code
            generic_results = analyze_generic_code(code, language)
            
            # Convert to expected format
            analysis_results = {
                'language': language,
                'analysis': {
                    'readability': f"Generic readability analysis - Maintainability: {generic_results['maintainability']['rating']}",
                    'performance': f"Complexity: {generic_results['complexity']['complexity_rating']} (Cyclomatic: {generic_results['complexity']['cyclomatic_complexity']})",
                    'security': "Generic security patterns checked - Consider language-specific security review",
                    'style': generic_results['style_issues'] if generic_results['style_issues'] else ["No major style issues detected"]
                },
                'summary': {
                    'language': language.title(),
                    'maintainability_score': generic_results['maintainability']['score'],
                    'complexity_rating': generic_results['complexity']['complexity_rating'],
                    'total_suggestions': len(generic_results['suggestions']),
                    'lines_analyzed': generic_results['metrics']['total_lines']
                },
                'suggestions': generic_results['suggestions'],
                'metrics': generic_results['metrics']
            }
            
            return analysis_results
    except Exception as e:
        # Fallback to basic analysis if specialized analyzers fail
        return {
            'language': language,
            'analysis': {
                'readability': f"Basic analysis only - Error in specialized analyzer: {str(e)[:100]}",
                'performance': "Basic complexity analysis performed",
                'security': "Basic security check performed"
            },
            'suggestions': [{'category': 'System', 'message': f'Analysis completed with limitations: {str(e)[:100]}'}],
            'summary': {'language': language, 'status': 'partial'}
        }

def handle_file_input():
    """Handle file input option with enhanced language detection."""
    filepath = styler.get_user_input("Enter file path")
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            return content, os.path.basename(filepath)
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin1') as file:
                content = file.read()
                print(format_message("File opened with latin1 encoding", 'warning'))
                return content, os.path.basename(filepath)
        except Exception as e:
            print(format_message(f"Error reading file: {str(e)}", 'error'))
            return None, None
    except Exception as e:
        print(format_message(f"Error reading file: {str(e)}", 'error'))
        return None, None

def save_analysis_results(code, analysis_results, corrected_code=None, filename=None):
    """Save analysis results to a file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cognito_analysis_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("COGNITO CODE ANALYSIS REPORT\n")
            file.write("=" * 50 + "\n\n")
            
            # Analysis metadata
            file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Language: {analysis_results.get('language', 'Unknown')}\n")
            file.write(f"Lines Analyzed: {len(code.split())}\n")
            file.write("\n" + "-" * 50 + "\n\n")
            
            # Original code
            file.write("ORIGINAL CODE:\n")
            file.write("-" * 20 + "\n")
            file.write(code)
            file.write("\n\n" + "-" * 50 + "\n\n")
            
            # Analysis results
            file.write("ANALYSIS RESULTS:\n")
            file.write("-" * 20 + "\n")
            
            if 'summary' in analysis_results:
                file.write("Summary:\n")
                for key, value in analysis_results['summary'].items():
                    file.write(f"  • {key.replace('_', ' ').title()}: {value}\n")
                file.write("\n")
            
            if 'suggestions' in analysis_results:
                file.write("Suggestions:\n")
                for suggestion in analysis_results['suggestions']:
                    if isinstance(suggestion, dict):
                        category = suggestion.get('category', 'General')
                        message = suggestion.get('message', '')
                        priority = suggestion.get('priority', 'medium')
                        file.write(f"  [{priority.upper()}] {category}: {message}\n")
                    else:
                        file.write(f"  • {suggestion}\n")
                file.write("\n")
            
            # Corrected code if available
            if corrected_code and corrected_code != code:
                file.write("-" * 50 + "\n\n")
                file.write("CORRECTED CODE:\n")
                file.write("-" * 20 + "\n")
                file.write(corrected_code)
                file.write("\n")
        
        print(format_message(f"Analysis saved to {filename}", 'success'))
        return True
    except Exception as e:
        print(format_message(f"Error saving analysis: {str(e)}", 'error'))
        return False

def process_suggestion_feedback(suggestion):
    """Get user feedback on a suggestion and update the learning system."""
    if not feedback_collector_available:
        return
        
    feedback = styler.get_user_input("Was this suggestion helpful?", 'choice').lower()
    accepted = feedback.startswith('y')
    
    if accepted:
        comment = styler.get_user_input("Any comments on why this was helpful? (Enter to skip)")
    else:
        comment = styler.get_user_input("Any comments on why this wasn't helpful? (Enter to skip)")
    
    try:
        feedback_collector = FeedbackCollector()
        feedback_collector.add_feedback(suggestion, accepted, comment if comment else None)
        
        # Show improvement metrics
        metrics = feedback_collector.get_metrics()
        print(format_message(f"Learning progress: {metrics['total_suggestions']} suggestions processed, "
                           f"{metrics['acceptance_rate']}% acceptance rate", 'info'))
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        print(format_message(f"Feedback recorded, but could not update learning system: {str(e)}", 'warning'))

def show_language_support():
    """Display comprehensive language support information."""
    styler.print_section_header("Language Support", 1)
    
    # Get support levels
    support_info = get_language_support_info()
    styler.print_language_support(support_info)
    
    # Show total count
    all_languages = get_supported_languages()
    print(format_message(f"Total languages supported: {len(all_languages)}", 'info'))
    
    # Show specific language info if requested
    show_details = styler.get_user_input("Show details for a specific language? (language name or 'no')")
    if show_details.lower() not in ['no', 'n', '']:
        lang_info = get_language_info(show_details.lower())
        if lang_info:
            styler.print_metrics_table(lang_info, f"{show_details.title()} Language Info")
        else:
            print(format_message(f"Language '{show_details}' not found in database", 'warning'))

def generate_improvement_report():
    """Generate and display improvement metrics report."""
    if not metrics_reporter_available:
        print(format_message("Metrics reporter module not available", 'error'))
        return
        
    try:
        reporter = ImprovementMetricsReporter()
        report_file = reporter.generate_improvement_report()
        print(format_message(f"Improvement metrics report generated: {report_file}", 'success'))
        
        # Display key metrics
        if feedback_collector_available:
            metrics = FeedbackCollector().get_metrics()
            if metrics["total_suggestions"] > 0:
                stats = {
                    'total_suggestions': metrics['total_suggestions'],
                    'acceptance_rate': f"{metrics['acceptance_rate']}%",
                    'learning_active': 'Yes' if metrics['total_suggestions'] > 10 else 'No'
                }
                styler.print_metrics_table(stats, "Key Performance Metrics")
            else:
                print(format_message("Not enough feedback data yet to generate metrics", 'warning'))
                print("Use Cognito more and provide feedback on suggestions to build up metrics.")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        print(format_message(f"Error generating report: {str(e)}", 'error'))

def analyze_with_progress(code, filename=None, language=None, use_llm=False, use_adaptive=False):
    """Analyze code with progress indication - FIXED RECURSION."""
    analysis_steps = [
        "Detecting language",
        "Analyzing readability", 
        "Checking performance",
        "Security analysis",
        "Generating suggestions"
    ]
    
    if use_llm:
        analysis_steps.extend(["AI enhancement", "Natural language processing"])
    
    print(format_message("Starting comprehensive code analysis", 'info'))
    
    # Progress through analysis steps
    for i, step in enumerate(analysis_steps):
        styler.print_progress(i + 1, len(analysis_steps), step)
        time.sleep(0.3)
    
    print()  # New line after progress
    
    # FIXED: Call analyze_code_enhanced directly, not recursively
    return analyze_code(code, filename, language, use_llm, use_adaptive)

def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cognito - Multi-language code analysis platform")
    parser.add_argument("--file", help="Path to code file to analyze")
    parser.add_argument("--language", help="Force language detection")
    parser.add_argument("--output", help="Output file for analysis results")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM enhancement")
    parser.add_argument("--adaptive", action="store_true", help="Use feedback-adaptive LLM mode")
    parser.add_argument("--report", action="store_true", help="Generate improvement metrics report")
    parser.add_argument("--languages", action="store_true", help="Show supported languages")
    parser.add_argument("--batch", help="Analyze multiple files in a directory")
    parser.add_argument("--version", action="version", version="Cognito v0.8.0")  
    args = parser.parse_args()
    
    # Handle specific command requests
    if args.report:
        generate_improvement_report()
        return
    
    if args.languages:
        show_language_support()
        return
    
    try:
        # Direct file analysis
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as file:
                    code_input = file.read()
                    filename = os.path.basename(args.file)
                    
                    print(format_message(f"Analyzing {filename}", 'info'))
                    
                    # Analyze with progress
                    analysis_results = analyze_with_progress(
                        code_input, filename, args.language, 
                        args.use_llm, args.adaptive
                    )
                    
                    # Print results
                    print_analysis_results(analysis_results, f"Analysis Results for {filename}")
                    
                    # Process feedback if available
                    if feedback_collector_available and analysis_results.get('suggestions'):
                        for suggestion in analysis_results['suggestions'][:3]:  # Limit to 3 for CLI
                            process_suggestion_feedback(suggestion)
                    
                    # Save results if requested
                    if args.output:
                        save_analysis_results(code_input, analysis_results, None, args.output)
                        
                    return
            except Exception as e:
                print(format_message(f"Error analyzing file: {str(e)}", 'error'))
                return
        
        # Batch analysis
        if args.batch:
            try:
                batch_dir = Path(args.batch)
                if not batch_dir.exists():
                    print(format_message(f"Directory not found: {args.batch}", 'error'))
                    return
                
                # Find code files
                code_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.go', '.rs', '.php', '.rb']
                code_files = []
                for ext in code_extensions:
                    code_files.extend(batch_dir.glob(f"**/*{ext}"))
                
                if not code_files:
                    print(format_message("No code files found in directory", 'warning'))
                    return
                
                print(format_message(f"Found {len(code_files)} code files for batch analysis", 'info'))
                
                # Analyze each file
                results_summary = {}
                for i, file_path in enumerate(code_files):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            code = file.read()
                            
                        print(f"\nanalyzing {file_path.name}...")
                        styler.print_progress(i + 1, len(code_files), f"Processing {file_path.name}")
                        
                        analysis = analyze_code(code, file_path.name, args.language, args.use_llm)
                        
                        # Store summary
                        results_summary[str(file_path)] = {
                            'language': analysis.get('language', 'unknown'),
                            'suggestions': len(analysis.get('suggestions', [])),
                            'maintainability': analysis.get('summary', {}).get('maintainability_score', 0)
                        }
                        
                    except Exception as e:
                        print(format_message(f"Error analyzing {file_path.name}: {str(e)}", 'error'))
                        continue
                
                # Print batch summary
                print("\n")
                styler.print_section_header("Batch Analysis Summary")
                styler.print_metrics_table(results_summary, "File Analysis Results")
                
                return
            except Exception as e:
                print(format_message(f"Error in batch analysis: {str(e)}", 'error'))
                return
        
        # Interactive mode
        styler.clear_screen()
        styler.print_logo()
        styler.print_separator()
        
        print(format_message("Welcome to Cognito! Multi-language code analysis platform.", 'info'))
        
        # Show supported languages
        supported = get_supported_languages()
        supported_str = ", ".join([lang.title() for lang in supported[:8]])  # Show first 8
        if len(supported) > 8:
            supported_str += f" and {len(supported) - 8} more"
        print(format_message(f"Supporting: {supported_str}", 'info'))
        
        while True:
            styler.print_separator()
            
            options = [
                "Analyze code snippet",
                "Load code from file", 
                "Batch analyze directory",
                "View language support",
                "View improvement metrics",
                "Exit"
            ]
            
            styler.print_menu(options, "Choose an option")
            choice = styler.get_user_input("Select option (1-6)", 'number')
            
            if choice == '1':
                print(format_message("Enter your code below (type 'DONE' on a new line when finished):", 'info'))
                print()
                code_lines = []
                while True:
                    line = input()
                    if line == 'DONE':
                        break
                    code_lines.append(line)
                
                code_input = '\n'.join(code_lines)
                if not code_input.strip():
                    print(format_message("No code entered", 'warning'))
                    continue
                
                filename = None
            
            elif choice == '2':
                code_input, filename = handle_file_input()
                if not code_input:
                    continue
            
            elif choice == '3':
                directory = styler.get_user_input("Enter directory path")
                # Use batch analysis logic from above
                continue
                
            elif choice == '4':
                show_language_support()
                continue
            
            elif choice == '5':
                generate_improvement_report()
                continue
                
            elif choice == '6':
                print(format_message("Thank you for using Cognito! Goodbye!", 'success'))
                sys.exit(0)
            
            else:
                print(format_message("Invalid choice. Please try again.", 'warning'))
                continue
            
            # Check for LLM enhancement
            use_llm = False
            use_adaptive = False
            
            if not args.use_llm:
                llm_choice = styler.get_user_input("Use AI enhancement?", 'choice').lower()
                use_llm = llm_choice.startswith('y')
                
                if use_llm and learning_llm_available:
                    adaptive_choice = styler.get_user_input("Use adaptive AI that learns from feedback?", 'choice').lower()
                    use_adaptive = adaptive_choice.startswith('y')
            else:
                use_llm = args.use_llm
                use_adaptive = args.adaptive
            
            # Analyze code with progress
            print()
            analysis_results = analyze_with_progress(code_input, filename, None, use_llm, use_adaptive)
            
            # Display results
            print_analysis_results(analysis_results)
            
            # Code correction if available
            if code_correction_available:
                styler.print_section_header("Code Improvement", 2)
                
                try:
                    # Extract issues for correction
                    feedback_items = {
                        'code': code_input,
                        'readability': analysis_results['analysis'].get('readability', ''),
                        'performance': analysis_results['analysis'].get('performance', ''),
                        'security': analysis_results['analysis'].get('security', '')
                    }
                    
                    issues = extract_issues_from_feedback(feedback_items)
                    
                    if issues:
                        language = analysis_results.get('language', 'unknown')
                        corrector = CodeCorrector(language)
                        corrected_code = corrector.correct_code(code_input, issues)
                        
                        if corrected_code != code_input:
                            print(format_message("Code improvements available!", 'success'))
                            
                            show_correction = styler.get_user_input("Show corrected code?", 'choice').lower()
                            if show_correction.startswith('y'):
                                styler.print_code_highlight(corrected_code)
                                
                                show_diff = styler.get_user_input("Show differences?", 'choice').lower()
                                if show_diff.startswith('y'):
                                    original_lines = code_input.split('\n')
                                    corrected_lines = corrected_code.split('\n')
                                    styler.print_diff(original_lines, corrected_lines)
                        else:
                            print(format_message("No automatic corrections available", 'info'))
                    else:
                        print(format_message("No issues found that can be automatically corrected", 'info'))
                        
                except Exception as e:
                    logger.error(f"Error in code correction: {e}")
                    print(format_message(f"Code correction error: {str(e)}", 'error'))
            
            # Process feedback
            if feedback_collector_available and analysis_results.get('suggestions'):
                styler.print_section_header("Feedback", 2)
                print(format_message("Help improve the system by providing feedback:", 'info'))
                
                for i, suggestion in enumerate(analysis_results['suggestions'][:3]):
                    print(f"\n{styler.SYMBOLS['subsection']} Suggestion {i+1}:")
                    print(format_suggestion(suggestion))
                    process_suggestion_feedback(suggestion)
            
            # Save option
            styler.print_separator('single')
            save_choice = styler.get_user_input("Save analysis results?", 'choice').lower()
            if save_choice.startswith('y'):
                custom_filename = styler.get_user_input("Enter filename (or press Enter for auto-generated)")
                save_analysis_results(code_input, analysis_results, None, custom_filename if custom_filename else None)

    except KeyboardInterrupt:
        print(f"\n{format_message('Exiting Cognito. Thank you for using our analysis platform!', 'success')}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(format_message(f"An unexpected error occurred: {str(e)}", 'error'))
        print(format_message("Check the logs for more details", 'warning'))
        sys.exit(1)

if __name__ == "__main__":
    check_recursion_limit()
    main()