"""
Clean visual styling for Cognito with professional symbols instead of emojis.
"""

from colorama import init, Fore, Style
import os

# Initialize colorama for cross-platform colored terminal output
init()

class CleanStyler:
    """Professional styling for terminal output."""
    
    # Clean symbols instead of emojis
    SYMBOLS = {
        'success': '✓',
        'error': '✗',
        'warning': '▲',
        'info': '►',
        'bullet': '•',
        'arrow_right': '→',
        'arrow_down': '↓',
        'section': '▶',
        'subsection': '‣',
        'separator': '─',
        'double_separator': '═',
        'corner': '└',
        'branch': '├',
        'vertical': '│',
        'analysis': '◊',
        'code': '◈',
        'suggestion': '◦',
        'priority_high': '■',
        'priority_medium': '▬',
        'priority_low': '▫'
    }
    
    # Color schemes
    COLORS = {
        'primary': Fore.CYAN,
        'success': Fore.GREEN,
        'error': Fore.RED,
        'warning': Fore.YELLOW,
        'info': Fore.BLUE,
        'muted': Fore.LIGHTBLACK_EX,
        'highlight': Fore.WHITE,
        'reset': Style.RESET_ALL
    }
    
    @classmethod
    def clear_screen(cls):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @classmethod
    def print_logo(cls):
        """Display the ASCII art logo for Cognito."""
        logo = f"""
{cls.COLORS['primary']}
 ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗████████╗ ██████╗ 
██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║╚══██╔══╝██╔═══██╗
██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║   ██║   ██║
██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║   ██║   ██║
╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║   ██║   ╚██████╔╝
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝    ╚═════╝ 
{cls.COLORS['reset']}
{cls.COLORS['success']}AI-Powered Code Analysis Platform{cls.COLORS['reset']} v0.7.1
"""
        print(logo)
    
    @classmethod
    def print_separator(cls, style='single'):
        """Print a separator line."""
        if style == 'double':
            print(f"{cls.COLORS['info']}{cls.SYMBOLS['double_separator']*70}{cls.COLORS['reset']}")
        else:
            print(f"{cls.COLORS['info']}{cls.SYMBOLS['separator']*70}{cls.COLORS['reset']}")
    
    @classmethod
    def print_section_header(cls, title, level=1):
        """Print a section header with formatting."""
        if level == 1:
            symbol = cls.SYMBOLS['section']
            color = cls.COLORS['primary']
        else:
            symbol = cls.SYMBOLS['subsection']
            color = cls.COLORS['info']
        
        print(f"\n{color}{symbol} {title}{cls.COLORS['reset']}")
        print(f"{color}{cls.SYMBOLS['separator']*50}{cls.COLORS['reset']}")
    
    @classmethod
    def format_status(cls, message, status='info'):
        """Format a status message with appropriate symbol and color."""
        symbol_map = {
            'success': cls.SYMBOLS['success'],
            'error': cls.SYMBOLS['error'],
            'warning': cls.SYMBOLS['warning'],
            'info': cls.SYMBOLS['info']
        }
        
        color_map = {
            'success': cls.COLORS['success'],
            'error': cls.COLORS['error'],
            'warning': cls.COLORS['warning'],
            'info': cls.COLORS['info']
        }
        
        symbol = symbol_map.get(status, cls.SYMBOLS['info'])
        color = color_map.get(status, cls.COLORS['info'])
        
        return f"{color}{symbol} {message}{cls.COLORS['reset']}"
    
    @classmethod
    def format_suggestion(cls, suggestion, category='general'):
        """Format a suggestion with appropriate styling."""
        if isinstance(suggestion, dict):
            priority = suggestion.get('priority', 'medium')
            message = suggestion.get('message', '')
            category = suggestion.get('category', category)
        else:
            priority = 'medium'
            message = str(suggestion)
        
        # Determine if this is a positive or negative message
        if cls._is_positive_message(message):
            return cls.format_status(message, 'success')
        else:
            # Format based on priority
            priority_symbols = {
                'high': cls.SYMBOLS['priority_high'],
                'medium': cls.SYMBOLS['priority_medium'],  
                'low': cls.SYMBOLS['priority_low']
            }
            
            priority_colors = {
                'high': cls.COLORS['error'],
                'medium': cls.COLORS['warning'],
                'low': cls.COLORS['info']
            }
            
            symbol = priority_symbols.get(priority, cls.SYMBOLS['priority_medium'])
            color = priority_colors.get(priority, cls.COLORS['warning'])
            
            return f"{color}{symbol} {message}{cls.COLORS['reset']}"
    
    @classmethod
    def _is_positive_message(cls, message):
        """Determine if a message represents a positive/good result."""
        if not isinstance(message, str):
            return False
        
        message_lower = message.lower()
        
        positive_patterns = [
            "looks good", "no issues detected", "no problems found",
            "passes", "security analysis: code passes", "owasp security checks",
            "code complexity looks good", "no style issues detected",
            "no common anti-patterns detected", "no common security issues detected",
            "code readability looks good", "excellent", "good readability"
        ]
        
        return any(pattern in message_lower for pattern in positive_patterns)
    
    @classmethod
    def print_metrics_table(cls, metrics, title="Analysis Metrics"):
        """Print metrics in a clean table format."""
        print(f"\n{cls.COLORS['primary']}{cls.SYMBOLS['analysis']} {title}{cls.COLORS['reset']}")
        print(f"{cls.COLORS['info']}{cls.SYMBOLS['separator']*50}{cls.COLORS['reset']}")
        
        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            elif isinstance(value, dict):
                value_str = f"{len(value)} items"
            else:
                value_str = str(value)
            
            print(f"{cls.COLORS['muted']}{cls.SYMBOLS['bullet']} {formatted_key}:{cls.COLORS['reset']} {value_str}")
    
    @classmethod
    def print_language_support(cls, supported_languages):
        """Print supported languages in a clean format."""
        print(f"\n{cls.COLORS['primary']}{cls.SYMBOLS['info']} Language Support{cls.COLORS['reset']}")
        print(f"{cls.COLORS['info']}{cls.SYMBOLS['separator']*30}{cls.COLORS['reset']}")
        
        if isinstance(supported_languages, dict):
            for category, languages in supported_languages.items():
                category_name = category.replace('_', ' ').title()
                print(f"{cls.COLORS['warning']}{cls.SYMBOLS['subsection']} {category_name}:{cls.COLORS['reset']}")
                
                if languages == ['*']:
                    print(f"  {cls.COLORS['muted']}{cls.SYMBOLS['bullet']} Any language (generic analysis){cls.COLORS['reset']}")
                else:
                    for lang in languages:
                        print(f"  {cls.COLORS['muted']}{cls.SYMBOLS['bullet']} {lang.title()}{cls.COLORS['reset']}")
                print()
        else:
            for lang in supported_languages:
                print(f"{cls.COLORS['muted']}{cls.SYMBOLS['bullet']} {lang.title()}{cls.COLORS['reset']}")
    
    @classmethod
    def print_progress(cls, current, total, description="Processing"):
        """Print a progress indicator."""
        percentage = (current / total) * 100 if total > 0 else 0
        filled = int(percentage // 4)  # 25 chars max
        bar = f"{'█' * filled}{'░' * (25 - filled)}"
        
        print(f"\r{cls.COLORS['info']}{cls.SYMBOLS['arrow_right']} {description}: {cls.COLORS['reset']}"
              f"[{cls.COLORS['primary']}{bar}{cls.COLORS['reset']}] "
              f"{percentage:.1f}% ({current}/{total})", end='', flush=True)
    
    @classmethod
    def print_completion_stats(cls, stats):
        """Print completion statistics."""
        print(f"\n\n{cls.COLORS['success']}{cls.SYMBOLS['success']} Analysis Complete{cls.COLORS['reset']}")
        print(f"{cls.COLORS['info']}{cls.SYMBOLS['separator']*30}{cls.COLORS['reset']}")
        
        for key, value in stats.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                if 'time' in key.lower():
                    value_str = f"{value:.1f}s"
                elif 'rate' in key.lower() or 'percentage' in key.lower():
                    value_str = f"{value:.1f}%"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            print(f"{cls.COLORS['muted']}{cls.SYMBOLS['bullet']} {formatted_key}: {cls.COLORS['reset']}{value_str}")
    
    @classmethod
    def get_user_input(cls, prompt, input_type='string'):
        """Get user input with clean formatting."""
        formatted_prompt = f"{cls.COLORS['primary']}{cls.SYMBOLS['arrow_right']} {prompt} {cls.COLORS['reset']}"
        
        if input_type == 'choice':
            formatted_prompt += f"{cls.COLORS['muted']}[y/n]: {cls.COLORS['reset']}"
        elif input_type == 'number':
            formatted_prompt += f"{cls.COLORS['muted']}(number): {cls.COLORS['reset']}"
        else:
            formatted_prompt += f"{cls.COLORS['muted']}: {cls.COLORS['reset']}"
        
        return input(formatted_prompt)
    
    @classmethod
    def print_menu(cls, options, title="Options"):
        """Print a clean menu."""
        print(f"\n{cls.COLORS['primary']}{cls.SYMBOLS['section']} {title}{cls.COLORS['reset']}")
        print(f"{cls.COLORS['info']}{cls.SYMBOLS['separator']*30}{cls.COLORS['reset']}")
        
        for i, option in enumerate(options, 1):
            print(f"{cls.COLORS['muted']}{i}. {cls.COLORS['reset']}{option}")
    
    @classmethod
    def print_code_highlight(cls, code, issues=None):
        """Print code with optional issue highlighting."""
        lines = code.split('\n')
        
        # Create issue mapping if provided
        issue_lines = {}
        if issues:
            for issue in issues:
                if hasattr(issue, 'get') and 'line' in issue:
                    line_num = issue.get('line', 0)
                    if line_num > 0 and line_num <= len(lines):
                        if line_num not in issue_lines:
                            issue_lines[line_num] = []
                        issue_lines[line_num].append(issue)
        
        print(f"{cls.COLORS['primary']}{cls.SYMBOLS['code']} Code Preview{cls.COLORS['reset']}")
        print(f"{cls.COLORS['info']}{cls.SYMBOLS['separator']*50}{cls.COLORS['reset']}")
        
        for i, line in enumerate(lines, 1):
            line_num_str = f"{i:4d}"
            
            if i in issue_lines:
                # Highlight line with issues
                print(f"{cls.COLORS['error']}{line_num_str} {cls.SYMBOLS['vertical']} {line}{cls.COLORS['reset']}")
                # Show issue descriptions
                for issue in issue_lines[i]:
                    if hasattr(issue, 'get'):
                        message = issue.get('message', '')
                        if message:
                            print(f"     {cls.SYMBOLS['corner']} {cls.COLORS['error']}{message}{cls.COLORS['reset']}")
            else:
                print(f"{cls.COLORS['muted']}{line_num_str} {cls.SYMBOLS['vertical']}{cls.COLORS['reset']} {line}")
    
    @classmethod
    def print_diff(cls, original_lines, modified_lines, title="Code Changes"):
        """Print a clean diff view."""
        print(f"\n{cls.COLORS['primary']}{cls.SYMBOLS['analysis']} {title}{cls.COLORS['reset']}")
        print(f"{cls.COLORS['info']}{cls.SYMBOLS['separator']*50}{cls.COLORS['reset']}")
        
        from difflib import unified_diff
        
        diff = unified_diff(
            original_lines, modified_lines,
            fromfile='original', tofile='modified',
            lineterm='', n=3
        )
        
        for line in diff:
            if line.startswith('---') or line.startswith('+++'):
                print(f"{cls.COLORS['info']}{line}{cls.COLORS['reset']}")
            elif line.startswith('@@'):
                print(f"{cls.COLORS['warning']}{line}{cls.COLORS['reset']}")
            elif line.startswith('-'):
                print(f"{cls.COLORS['error']}{cls.SYMBOLS['error']} {line[1:]}{cls.COLORS['reset']}")
            elif line.startswith('+'):
                print(f"{cls.COLORS['success']}{cls.SYMBOLS['success']} {line[1:]}{cls.COLORS['reset']}")
            else:
                print(f"{cls.COLORS['muted']}  {line}{cls.COLORS['reset']}")


def format_message(message, status='info'):
    """Convenience function for formatting messages."""
    return CleanStyler.format_status(message, status)


def format_suggestion(suggestion, category='general'):
    """Convenience function for formatting suggestions."""
    return CleanStyler.format_suggestion(suggestion, category)


def print_analysis_results(results, title="Analysis Results"):
    """Print comprehensive analysis results with clean formatting."""
    styler = CleanStyler()
    
    styler.print_section_header(title)
    
    # Print summary metrics if available
    if 'metrics' in results:
        styler.print_metrics_table(results['metrics'], "Code Metrics")
    
    # Print suggestions
    if 'suggestions' in results and results['suggestions']:
        print(f"\n{styler.COLORS['primary']}{styler.SYMBOLS['section']} Suggestions{styler.COLORS['reset']}")
        print(f"{styler.COLORS['info']}{styler.SYMBOLS['separator']*50}{styler.COLORS['reset']}")
        
        for suggestion in results['suggestions']:
            formatted = styler.format_suggestion(suggestion)
            print(formatted)
    
    # Print language info if available
    if 'language' in results:
        language = results['language']
        confidence = results.get('confidence', 0)
        print(f"\n{styler.COLORS['info']}{styler.SYMBOLS['info']} Detected Language: {styler.COLORS['reset']}"
              f"{language.title()} ({confidence:.1f}% confidence)")
    
    return styler