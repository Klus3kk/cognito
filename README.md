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

# Install dependencies
cd cognito
pip install -r requirements.txt
```

## Usage
**Cognito** can be run by inputting code directly into the application, where it will provide real-time analysis and suggestions.

```bash
# Run the main application
python src/main.py
```

## Progress
Work's in **REALLY REALLY EARLY** process.
