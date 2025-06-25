# Cognito

**Cognito** is an interactive AI assistant that provides code review insights one suggestion at a time. It supports multiple programming languages (Python, C, C++...) and uses ML/NLP and LLM technologies to enhance code quality by:

- **Improving readability**
- **Ensuring security** (aligned with OWASP guidelines)
- **Optimizing code performance**
- **Providing AI-powered insights**

## Key Features

**Multi-Language Analysis**
   - Supports Python, C, C++, Java, JavaScript...
   - Universal analysis engine with language-specific optimizations
   - Intelligent language detection with confidence scoring
   
**Code Readability Analysis**
   - Checks for naming conventions, comments, and code structure
   - Suggests refactoring for clearer and more maintainable code
   - Automated code correction with pattern recognition
     
**Security Vulnerability Detection**
   - Detects common security issues and flags vulnerabilities according to OWASP standards
   - Language-specific security checks (buffer overflows, SQL injection, XSS)
   - Highlights risky functions and suggests safer alternatives
     
**Performance Analysis**
   - Analyzes algorithmic complexity and memory usage
   - Flags inefficient patterns and recommends optimizations
   - Big O complexity detection and optimization suggestions
     
**Interactive Suggestions**
   - Provides one suggestion at a time, allowing users to review, accept, or dismiss
   - Feedback collection system that improves suggestions over time
   - Offers optional explanations for each suggestion if requested
     
**AI-Powered Analysis**
   - Uses LLM technology to provide natural language explanations of code
   - Enhances suggestions with AI-driven insights and alternatives
   - Adaptive learning system that improves based on user feedback
   - Custom ML model training with HuggingFace integration

## Installation
```bash
git clone https://github.com/Klus3kk/cognito.git

cd cognito

# Automated installer (recommended)
./install.sh  # Linux/macOS
# or
.\install.ps1  # Windows

# OR

# Manual installation
pip install -e .

# For ML features (recommended)
# Generate a token on Huggingface
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

# Batch analysis of entire directories
cognito --batch /path/to/project/

# Save analysis output to a file
cognito --file path/to/your/code.py --output analysis_results.txt

# Generate improvement metrics report
cognito --metrics
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
