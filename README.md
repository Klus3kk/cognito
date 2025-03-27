# Cognito

## Overview
**Cognito** is an interactive AI assistant that provides code review insights one suggestion at a time. It initially supports Python and C, with plans to add more languages. By leveraging ML/NLP and LLM technologies, this project aims to enhance code quality by:
- **Improving readability**
- **Ensuring security** (aligned with OWASP guidelines)
- **Optimizing code performance**
- **Providing AI-powered insights**

## Key Features
* **Code Readability Analysis**
   - Checks for naming conventions, comments, and code structure.
   - Suggests refactoring for clearer and more maintainable code.
* **Security Vulnerability Detection**
   - Detects common security issues and flags vulnerabilities according to OWASP standards.
   - Highlights risky functions and suggests safer alternatives.
* **Performance Analysis**
   - Analyzes algorithmic complexity and memory usage.
   - Flags inefficient patterns and recommends optimizations.
* **Interactive Suggestions**
   - Provides one suggestion at a time, allowing users to review, accept, or dismiss.
   - Offers optional explanations for each suggestion if requested.
* **AI-Powered Analysis**
   - Uses LLM technology to provide natural language explanations of code
   - Enhances suggestions with AI-driven insights and alternatives
   - Identifies semantic patterns that rule-based analysis might miss

## Installation
```bash
# Clone the repository
git clone https://github.com/Klus3kk/cognito.git

# Navigate to repository
cd cognito

# Install the package from the local directory
pip install -e .

# For ML features (recommended)
# Create an account at huggingface.co and generate a token
export HUGGINGFACE_TOKEN="your_token_here"

# For AI-powered analysis (optional)
# Get an API key from OpenAI
export OPENAI_API_KEY="your_openai_api_key"
```

## Usage
After installation, you can run **Cognito** using the command-line interface:

```bash
# Run with standard analysis
cognito

# Run with AI-powered analysis
cognito --use-llm

# Analyze a specific file
cognito --file path/to/your/code.py

# Analyze a file with AI enhancement
cognito --file path/to/your/code.py --use-llm

# Specify the language manually
cognito --file path/to/your/code.c --language c

# Save analysis output to a file
cognito --file path/to/your/code.py --output analysis_results.txt
```

### Docker Support
Cognito can also be run using Docker for a consistent environment:

```bash
# Build the Docker image
docker build -t cognito .

# Run Cognito with Docker
docker run -it -e OPENAI_API_KEY=your_key_here cognito

# Analyze a specific file with Docker
docker run -it -v $(pwd):/app/code cognito python -m src.main --file /app/code/your_file.py --use-llm
```
