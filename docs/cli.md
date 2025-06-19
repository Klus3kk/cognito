# Command Line Interface

The Cognito CLI provides comprehensive code analysis with multiple options and modes.

## Basic Usage

```bash
# Interactive mode - enter code directly
cognito

# Analyze a specific file
cognito --file path/to/code.py

# Analyze with AI enhancement
cognito --file code.py --use-llm

# Batch analysis of directory
cognito --batch /path/to/project/
```

## Command Options

### File Analysis
```bash
# Analyze single file
cognito --file <filepath>

# Specify language manually
cognito --file code.txt --language python

# Save analysis results
cognito --file code.py --output results.txt
```

### Analysis Modes
```bash
# Standard analysis (default)
cognito --file code.py

# AI-enhanced analysis
cognito --file code.py --use-llm

# Adaptive learning mode
cognito --file code.py --adaptive

# Batch processing
cognito --batch /path/to/directory/
```

### Output Options
```bash
# Save to file
cognito --file code.py --output analysis.txt

# JSON format output
cognito --file code.py --format json

# Verbose output
cognito --file code.py --verbose

# Quiet mode (minimal output)
cognito --file code.py --quiet
```

### Information Commands
```bash
# Show version
cognito --version

# List supported languages
cognito --languages

# Show language details
cognito --language-info python

# Generate metrics report
cognito --metrics

# Show feedback statistics
cognito --feedback-stats
```

## Interactive Mode

When run without arguments, Cognito enters interactive mode:

```bash
$ cognito
```

You can then:
1. Paste or type code directly
2. Press Ctrl+D (Linux/macOS) or Ctrl+Z (Windows) to analyze
3. Review suggestions one by one
4. Accept or reject each suggestion

## Language Detection

Cognito automatically detects programming languages:

```bash
# Automatic detection
cognito --file unknown_file.txt

# Manual specification
cognito --file script --language python

# Show detection confidence
cognito --file code.py --verbose
```

Supported languages:
- Python (.py)
- C (.c, .h)
- C++ (.cpp, .cxx, .hpp)
- Java (.java)
- JavaScript (.js, .jsx)
- And more with universal analysis

## Analysis Types

### Security Analysis
Cognito performs OWASP-compliant security analysis:

```bash
# Standard security checks included
cognito --file webapp.py

# Focus on security issues
cognito --file code.py --priority security
```

Common security checks:
- SQL injection vulnerabilities
- XSS prevention
- Buffer overflow detection
- Unsafe function usage
- Input validation issues

### Performance Analysis
Identifies performance bottlenecks:

```bash
# Performance analysis included by default
cognito --file algorithm.py
```

Performance checks:
- Algorithmic complexity (Big O)
- Memory usage patterns
- Inefficient loops
- Resource management
- Optimization opportunities

### Readability Analysis
Evaluates code readability and maintainability:

```bash
# Readability analysis always included
cognito --file messy_code.py
```

Readability factors:
- Naming conventions
- Code structure
- Documentation coverage
- Function complexity
- Indentation consistency

## AI Features

### LLM Integration
Enhanced analysis with OpenAI:

```bash
# Requires OPENAI_API_KEY environment variable
cognito --file code.py --use-llm
```

AI capabilities:
- Natural language explanations
- Context-aware suggestions
- Code improvement recommendations
- Educational insights

### Adaptive Learning
Cognito learns from your feedback:

```bash
# Enable adaptive mode
cognito --file code.py --adaptive
```

Learning features:
- Suggestion quality improvement
- User preference adaptation
- Category-specific optimization
- Feedback-driven enhancement

## Batch Processing

Analyze entire projects or directories:

```bash
# Analyze all code files in directory
cognito --batch /path/to/project/

# Specific file extensions
cognito --batch /path/to/project/ --extensions .py,.js

# Recursive analysis
cognito --batch /path/to/project/ --recursive

# Generate summary report
cognito --batch /path/to/project/ --output batch_report.txt
```

## Configuration

### Environment Variables
```bash
# AI features
export OPENAI_API_KEY="your_key"
export HUGGINGFACE_TOKEN="your_token"

# Behavior configuration
export COGNITO_LOG_LEVEL="INFO"
export COGNITO_MAX_FILE_SIZE="1048576"
export COGNITO_TIMEOUT_SECONDS="30"
```

### Configuration Files
Cognito looks for configuration in:
- `~/.cognito/config.yaml`
- `./cognito.yaml`
- Environment variables

## Output Formats

### Standard Output
Default human-readable format with colors and styling.

### JSON Output
Machine-readable format for integration:

```bash
cognito --file code.py --format json
```

Example JSON output:
```json
{
  "file": "code.py",
  "language": "python",
  "confidence": 95.0,
  "suggestions": [
    {
      "category": "Style",
      "message": "Function name should use snake_case",
      "priority": "medium",
      "line": 5
    }
  ],
  "metrics": {
    "readability_score": 4,
    "complexity": "low",
    "security_issues": 0
  }
}
```

### File Output
Save results to file:

```bash
# Text format
cognito --file code.py --output results.txt

# JSON format
cognito --file code.py --format json --output results.json
```
