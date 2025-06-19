# Installation Guide

## Automated Installation (Recommended)

Cognito provides platform-specific installers that handle all dependencies and configuration automatically.

### Linux/macOS
```bash
git clone https://github.com/Klus3kk/cognito.git
cd cognito
./install.sh
```

The installer will:
- Detect your system and package manager
- Install Python dependencies
- Set up virtual environment
- Configure environment variables
- Optionally install Docker support

### Windows
```powershell
git clone https://github.com/Klus3kk/cognito.git
cd cognito
.\install.ps1
```

The PowerShell installer will:
- Check Windows compatibility
- Validate Python installation
- Create virtual environment
- Install dependencies
- Set up configuration

## Installation Types (planned functionality in the future c:)

The installer offers several installation types:

**Standard** - Full installation with all features
- Core analysis engine
- All language analyzers
- AI/LLM capabilities
- Feedback system

**Minimal** - Core features only
- Basic analysis without AI features
- Reduced dependencies
- Faster installation

**Development** - Full installation + dev tools
- All standard features
- Testing framework
- Development dependencies
- Pre-commit hooks

**Docker** - Containerized installation
- Docker image creation
- Docker Compose setup
- Isolated environment

## Manual Installation

If you prefer manual installation:

```bash
# Clone repository
git clone https://github.com/Klus3kk/cognito.git
cd cognito

# Create virtual environment
python -m venv cognito-env
source cognito-env/bin/activate  # Linux/macOS
# or
cognito-env\Scripts\activate     # Windows

# Install dependencies
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev]"
```

## Environment Configuration

### Required Environment Variables

```bash
# For AI features (optional)
export OPENAI_API_KEY="your_openai_api_key"

# For ML model training (optional)
export HUGGINGFACE_TOKEN="your_huggingface_token"
```

### Optional Configuration

```bash
# Custom installation directory
export COGNITO_INSTALL_DIR="/custom/path"

# Log level
export COGNITO_LOG_LEVEL="INFO"

# Disable telemetry
export COGNITO_ENABLE_TELEMETRY="false"
```

## Docker Installation

### Build from Source
```bash
git clone https://github.com/Klus3kk/cognito.git
cd cognito
docker build -t cognito .
```

### Run with Docker
```bash
# Interactive mode
docker run -it cognito

# Analyze specific file
docker run -it -v $(pwd):/app/code cognito \
  python -m src.main --file /app/code/example.py

# With AI features
docker run -it -e OPENAI_API_KEY=your_key cognito \
  python -m src.main --file /app/code/example.py --use-llm
```

### Docker Compose
```bash
# If installed with Docker support
cd ~/.cognito  # or your installation directory
docker-compose up -d
```

## Verification

After installation, verify Cognito is working:

```bash
# Check installation
cognito --version

# Test basic analysis
echo "def hello(): print('world')" | cognito

# Test with a file
echo "def calculate_sum(a, b): return a + b" > test.py
cognito --file test.py
rm test.py
```

## Requirements

### System Requirements
- **Python**: 3.8 - 3.12 (3.10 would be the best)
- **Disk**: 2-3GB free space

## Troubleshooting Installation

### Common Issues

**Python Version Issues**
```bash
# Check Python version
python --version

# If using multiple Python versions
python3.8 -m pip install -e .
```

**Permission Issues (Linux/macOS)**
```bash
# Don't use sudo with the installer
# Use --user flag if needed
pip install --user -e .
```

**Windows PowerShell Execution Policy**
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run installer
.\install.ps1
```

**Network Issues**
```bash
# Use alternative index
pip install -e . --index-url https://pypi.org/simple/

# Behind corporate firewall
pip install -e . --trusted-host pypi.org --trusted-host pypi.python.org
```

### Uninstallation

**Remove Cognito**
```bash
# If installed with installer
cd ~/.cognito  # or installation directory
./uninstall.sh  # Linux/macOS
# or
.\uninstall.ps1  # Windows

# Manual removal
pip uninstall cognito
rm -rf ~/.cognito
```

**Clean Docker Installation**
```bash
docker rmi cognito:latest
docker system prune -f
```

## Next Steps

After installation:
1. Read the [Quick Start Guide](quickstart.md)
2. Configure your [environment](configuration.md)
3. Explore [language support](languages.md)
4. Set up [AI features](ai-features.md)
