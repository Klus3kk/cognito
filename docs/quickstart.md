# Quick Start Guide

Get up and running with Cognito in just a few minutes.

## Installation

### Option 1: Automated Installer (Recommended)
```bash
git clone https://github.com/Klus3kk/cognito.git
cd cognito
./install.sh  # Linux/macOS
# or
.\install.ps1  # Windows
```

### Option 2: Manual Installation
```bash
git clone https://github.com/Klus3kk/cognito.git
cd cognito
pip install -e .
```

## First Analysis

### Interactive Mode
Start Cognito and paste your code:
```bash
$ cognito
```

### File Analysis
Analyze a specific file:
```bash
# Create a test file
echo "def hello(): print('world')" > example.py

# Analyze it
cognito --file example.py
```

## AI-Enhanced Analysis

Enable AI features for deeper insights:

### 1. Set Up OpenAI API
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### 2. Run AI Analysis
```bash
cognito --file demo.py --use-llm
```

## Batch Analysis

Analyze entire projects:

### 1. Create Project Structure
```bash
mkdir myproject
cd myproject

# Create some Python files
echo "def func1(): pass" > file1.py
echo "def func2(): pass" > file2.py
echo "class MyClass: pass" > file3.py
```

### 2. Run Batch Analysis
```bash
cognito --batch .
```

## Configuration

### Basic Configuration
Create a config file to customize behavior:

```bash
# Create config directory
mkdir -p ~/.cognito

# Create basic config
cat > ~/.cognito/config.yaml << 'EOF'
analysis:
  max_line_length: 88  # Black-style instead of PEP 8's 79
  ignore_docstring_warnings: false
  
security:
  strict_mode: true
  
performance:
  suggest_optimizations: true
  
output:
  style: "clean"  # or "detailed", "minimal"
EOF
```

### Environment Variables
```bash
# Optional: Enable ML features
export HUGGINGFACE_TOKEN="your_token"

# Optional: Customize behavior
export COGNITO_LOG_LEVEL="INFO"
export COGNITO_MAX_FILE_SIZE="1048576"  # 1MB
```

## Common Use Cases

### 1. Code Review Before Commit
```bash
# Analyze changed files
git diff --name-only | grep '\.py$' | xargs cognito --file

# Or analyze all Python files
find . -name "*.py" -exec cognito --file {} \;
```

### 2. Learning Mode
```bash
# Get explanations for all suggestions
cognito --file code.py --explain-all

# AI-enhanced learning
cognito --file code.py --use-llm --verbose
```

### 3. Security Audit
```bash
# Focus on security issues
cognito --file webapp.py --priority security

# Batch security scan
cognito --batch . --priority security --output security_report.txt
```

### 4. Performance Optimization
```bash
# Get performance suggestions
cognito --file algorithm.py --priority performance
```

## Understanding Output

### Suggestion Categories
- **Security**: Vulnerability detection and safe coding practices
- **Performance**: Optimization opportunities and efficiency improvements
- **Style**: Code formatting and convention compliance
- **Readability**: Code clarity and maintainability
- **Documentation**: Missing or inadequate documentation

### Priority Levels
- **High**: Critical security issues, major performance problems
- **Medium**: Important style issues, moderate performance gains
- **Low**: Minor improvements, documentation suggestions

### Confidence Scores
For language detection and AI suggestions:
- **90-100%**: High confidence
- **70-89%**: Good confidence
- **50-69%**: Moderate confidence
- **Below 50%**: Low confidence (may be inaccurate)
