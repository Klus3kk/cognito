# Cognito

## Overview
**Cognito** is an interactive AI assistant that provides code review insights one suggestion at a time. It initially supports Python and C, with plans to add more languages. By leveraging ML/NLP, **Cognito** aims to enhance code quality by:
- **Improving readability**
- **Ensuring security** (aligned with OWASP guidelines)
- **Optimizing code performance**

## Key Features
1. **Code Readability Analysis**
   - Checks for naming conventions, comments, and code structure.
   - Suggests refactoring for clearer and more maintainable code.
2. **Security Vulnerability Detection**
   - Detects common security issues and flags vulnerabilities according to OWASP standards.
   - Highlights risky functions and suggests safer alternatives.
3. **Performance Analysis**
   - Analyzes algorithmic complexity and memory usage.
   - Flags inefficient patterns and recommends optimizations.
4. **Interactive Suggestions**
   - Provides one suggestion at a time, allowing users to review, accept, or dismiss.
   - Offers optional explanations for each suggestion if requested.

## Installation
```bash
# Clone the repository
git clone https://github.com/Klus3kk/cognito.git

# Navigate to repository
cd cognito

# Install the package from the local directory
pip install -e .

# Set up Hugging Face token for ML features (required)
# Create an account at huggingface.co and generate a token
export HUGGINGFACE_TOKEN="your_token_here"
```

## Usage
After installation, you can run **Cognito** using the command-line interface:

```bash
# Simply run the command
cognito
```

### How It Works
1. When you start Cognito, you'll see a menu with options to:
   - Enter code directly
   - Load code from a file
   - Exit the program

2. After entering or loading code, Cognito will:
   - Automatically detect the programming language
   - Analyze the code for readability issues using ML models
   - Check for performance bottlenecks and complexity issues
   - Scan for potential security vulnerabilities
   - Display the results with color-coded feedback

3. You can then save the analysis results to a file for future reference.

## Features in Detail

### Language Detection
Cognito automatically identifies whether you're working with Python or C code, allowing for language-specific analysis without manual configuration.

### ML-Powered Analysis
Using the CodeBERT model from Hugging Face, Cognito provides intelligent code readability assessment that goes beyond simple rule-based checking.

### Security Assessment
The security analyzer identifies potential vulnerabilities based on OWASP guidelines, including:
- Injection vulnerabilities
- Insecure function usage
- Hardcoded credentials
- Path traversal risks

### Performance Insights
The performance analyzer evaluates code efficiency by examining:
- Algorithmic complexity
- Nested loop structures
- Memory usage patterns
- Recursive function safety

### Interactive Interface
The color-coded terminal interface makes it easy to:
- Identify critical issues (marked in red)
- Celebrate good practices (marked in green)
- Save comprehensive reports for future reference

## Progress
Work's in **REALLY REALLY EARLY** process, but the core functionality is operational. Future development will focus on expanding language support, improving ML model accuracy, and adding IDE integrations.