#!/bin/bash

# Cognito AI Code Analysis Platform - Automated Installer
# Version: 1.0.0
# Usage: curl -sSL https://raw.githubusercontent.com/yourusername/cognito/main/install.sh | bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
COGNITO_VERSION="0.8.0"
INSTALL_DIR="${COGNITO_INSTALL_DIR:-$HOME/.cognito}"
PYTHON_MIN_VERSION="3.8"
PYTHON_MAX_VERSION="3.12"
VENV_NAME="cognito-env"

# Platform detection
OS=""
ARCH=""
PACKAGE_MANAGER=""

# Installation options
INSTALL_TYPE="standard"  # standard, development, docker, minimal
INSTALL_DOCKER=false
INSTALL_NVIDIA=false
SETUP_SYSTEMD=false
CONFIGURE_NGINX=false
ENABLE_LLM=false
INSTALL_OPTIONAL_DEPS=false

# ASCII Art Logo
show_logo() {
    echo -e "${CYAN}"
    cat << 'EOF'
 ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗████████╗ ██████╗ 
██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║╚══██╔══╝██╔═══██╗
██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║   ██║   ██║
██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║   ██║   ██║
╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║   ██║   ╚██████╔╝
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝    ╚═════╝ 
                                                           
    AI-Powered Code Analysis Platform - Installer v1.0
EOF
    echo -e "${NC}"
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${PURPLE}[STEP]${NC} $1"
}

# Progress bar
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))
    
    printf "\r${CYAN}Progress: [${NC}"
    printf "%${completed}s" | tr ' ' '█'
    printf "%${remaining}s" | tr ' ' '░'
    printf "${CYAN}] %d%% (%d/%d)${NC}" $percentage $current $total
    
    if [ $current -eq $total ]; then
        echo ""
    fi
}

# Detect operating system and architecture
detect_platform() {
    log_step "Detecting platform"
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &> /dev/null; then
            PACKAGE_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            PACKAGE_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            PACKAGE_MANAGER="dnf"
        elif command -v pacman &> /dev/null; then
            PACKAGE_MANAGER="pacman"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        if command -v brew &> /dev/null; then
            PACKAGE_MANAGER="brew"
        fi
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    fi
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64|amd64) ARCH="x64" ;;
        i386|i686) ARCH="x86" ;;
        aarch64|arm64) ARCH="arm64" ;;
        armv7l) ARCH="arm" ;;
        *) ARCH="unknown" ;;
    esac
    
    log_info "Detected platform: $OS ($ARCH)"
    log_info "Package manager: $PACKAGE_MANAGER"
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        log_info "Found Python $PYTHON_VERSION"
        
        # Version comparison
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) and sys.version_info < (3, 13) else 1)"; then
            log_success "Python version is compatible"
        else
            log_error "Python version $PYTHON_VERSION is not supported. Please install Python 3.8-3.12"
            exit 1
        fi
    else
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed"
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        log_warn "Git is not installed. Some features may not work"
    fi
    
    # Check available disk space (minimum 1GB)
    if command -v df &> /dev/null; then
        available_space=$(df "$HOME" | awk 'NR==2 {print $4}')
        if [ "$available_space" -lt 1048576 ]; then  # 1GB in KB
            log_warn "Less than 1GB disk space available"
        fi
    fi
    
    # Check memory (minimum 2GB)
    if command -v free &> /dev/null; then
        total_mem=$(free -m | awk 'NR==2{print $2}')
        if [ "$total_mem" -lt 2048 ]; then
            log_warn "Less than 2GB RAM available. Some features may be slower"
        fi
    fi
}

# Interactive configuration
configure_installation() {
    log_step "Configuring installation"
    
    echo "Please select installation type:"
    echo "1) Standard - Full installation with all features"
    echo "2) Minimal - Core features only"
    echo "3) Development - Full installation + development tools"
    echo "4) Docker - Docker-based installation"
    
    read -p "Enter choice [1-4] (default: 1): " choice
    case $choice in
        2) INSTALL_TYPE="minimal" ;;
        3) INSTALL_TYPE="development" ;;
        4) INSTALL_TYPE="docker" ;;
        *) INSTALL_TYPE="standard" ;;
    esac
    
    log_info "Selected installation type: $INSTALL_TYPE"
    
    # Ask about LLM features
    read -p "Enable LLM/AI features? (requires OpenAI API key) [y/N]: " enable_llm
    if [[ $enable_llm =~ ^[Yy]$ ]]; then
        ENABLE_LLM=true
        echo "You'll need to configure your OpenAI API key after installation"
    fi
    
    # Ask about optional dependencies
    if [[ "$INSTALL_TYPE" == "standard" ]] || [[ "$INSTALL_TYPE" == "development" ]]; then
        read -p "Install optional ML dependencies? (TensorFlow, etc.) [y/N]: " install_optional
        if [[ $install_optional =~ ^[Yy]$ ]]; then
            INSTALL_OPTIONAL_DEPS=true
        fi
    fi
    
    # Ask about Docker
    if command -v docker &> /dev/null && [[ "$INSTALL_TYPE" != "docker" ]]; then
        read -p "Also install Docker images? [y/N]: " install_docker
        if [[ $install_docker =~ ^[Yy]$ ]]; then
            INSTALL_DOCKER=true
        fi
    fi
    
    # Ask about system service (Linux only)
    if [[ "$OS" == "linux" ]] && [[ $EUID -eq 0 ]] && [[ "$INSTALL_TYPE" != "minimal" ]]; then
        read -p "Install as system service? [y/N]: " setup_service
        if [[ $setup_service =~ ^[Yy]$ ]]; then
            SETUP_SYSTEMD=true
        fi
    fi
    
    echo ""
    log_info "Installation directory: $INSTALL_DIR"
    read -p "Change installation directory? [y/N]: " change_dir
    if [[ $change_dir =~ ^[Yy]$ ]]; then
        read -p "Enter new path: " new_dir
        INSTALL_DIR="$new_dir"
    fi
}

# Install system dependencies
install_system_deps() {
    log_step "Installing system dependencies"
    
    case $PACKAGE_MANAGER in
        apt)
            log_info "Updating package list..."
            sudo apt-get update -qq
            
            log_info "Installing system packages..."
            sudo apt-get install -y \
                python3-pip python3-venv python3-dev \
                build-essential git curl wget \
                libssl-dev libffi-dev \
                pkg-config
            ;;
        yum|dnf)
            log_info "Installing system packages..."
            sudo $PACKAGE_MANAGER install -y \
                python3-pip python3-devel \
                gcc gcc-c++ make git curl wget \
                openssl-devel libffi-devel \
                pkgconfig
            ;;
        brew)
            log_info "Installing system packages..."
            brew install python git curl wget
            ;;
        pacman)
            log_info "Installing system packages..."
            sudo pacman -S --noconfirm \
                python python-pip git curl wget \
                base-devel openssl libffi \
                pkgconf
            ;;
        *)
            log_warn "Unknown package manager. Please install Python 3, pip, git, and build tools manually"
            ;;
    esac
}

# Create installation directory and virtual environment
setup_environment() {
    log_step "Setting up environment"
    
    # Create installation directory
    log_info "Creating installation directory: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Create virtual environment
    log_info "Creating Python virtual environment..."
    python3 -m venv "$VENV_NAME"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    log_success "Environment setup complete"
}

# Download and install Cognito
install_cognito() {
    log_step "Installing Cognito"
    
    # Download source code
    log_info "Downloading Cognito source code..."
    if command -v git &> /dev/null; then
        git clone https://github.com/yourusername/cognito.git .
        git checkout main
    else
        log_info "Downloading release archive..."
        curl -L "https://github.com/yourusername/cognito/archive/main.tar.gz" -o cognito.tar.gz
        tar -xzf cognito.tar.gz --strip-components=1
        rm cognito.tar.gz
    fi
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    source "$VENV_NAME/bin/activate"
    
    case $INSTALL_TYPE in
        minimal)
            # Create minimal requirements
            cat > requirements-minimal.txt << 'EOF'
pytest>=7.4.0
colorama>=0.4.6
astroid>=3.0.1
EOF
            pip install -r requirements-minimal.txt
            ;;
        development)
            pip install -r requirements.txt
            pip install -e ".[dev]"
            ;;
        *)
            pip install -r requirements.txt
            pip install -e .
            ;;
    esac
    
    # Install optional dependencies if requested
    if [[ "$INSTALL_OPTIONAL_DEPS" == true ]]; then
        log_info "Installing optional ML dependencies..."
        pip install -e ".[full]"
    fi
    
    log_success "Cognito installation complete"
}

# Create configuration files
setup_configuration() {
    log_step "Setting up configuration"
    
    # Create configuration directory
    mkdir -p "$INSTALL_DIR/config"
    mkdir -p "$INSTALL_DIR/data"
    mkdir -p "$INSTALL_DIR/logs"
    mkdir -p "$INSTALL_DIR/models"
    
    # Create environment file
    cat > "$INSTALL_DIR/config/.env" << EOF
# Cognito Configuration
COGNITO_ENV=production
COGNITO_DEBUG=false
COGNITO_VERSION=$COGNITO_VERSION

# Installation paths
COGNITO_INSTALL_DIR=$INSTALL_DIR
COGNITO_DATA_DIR=$INSTALL_DIR/data
COGNITO_LOG_FILE=$INSTALL_DIR/logs/cognito.log
COGNITO_MODELS_DIR=$INSTALL_DIR/models
COGNITO_CACHE_DIR=$INSTALL_DIR/data/cache

# Security settings
COGNITO_RATE_LIMIT=1000
COGNITO_MAX_FILE_SIZE_MB=10
COGNITO_ENABLE_SECURITY_VALIDATION=true

# Features
COGNITO_ENABLE_LLM=$ENABLE_LLM
COGNITO_ENABLE_CODE_CORRECTION=true
COGNITO_ENABLE_FEEDBACK_LEARNING=true

# API Keys (set these manually)
# OPENAI_API_KEY=your_openai_api_key_here
# HUGGINGFACE_TOKEN=your_huggingface_token_here
EOF
    
    # Create launcher script
    cat > "$INSTALL_DIR/cognito" << EOF
#!/bin/bash
# Cognito Launcher Script

COGNITO_DIR="$INSTALL_DIR"
VENV_DIR="\$COGNITO_DIR/$VENV_NAME"

# Load environment
if [ -f "\$COGNITO_DIR/config/.env" ]; then
    source "\$COGNITO_DIR/config/.env"
fi

# Activate virtual environment
source "\$VENV_DIR/bin/activate"

# Change to Cognito directory
cd "\$COGNITO_DIR"

# Run Cognito
python -m src.main "\$@"
EOF
    
    chmod +x "$INSTALL_DIR/cognito"
    
    # Create system-wide launcher if possible
    if [[ -w "/usr/local/bin" ]] || [[ $EUID -eq 0 ]]; then
        log_info "Creating system-wide launcher..."
        cat > "/usr/local/bin/cognito" << EOF
#!/bin/bash
exec "$INSTALL_DIR/cognito" "\$@"
EOF
        chmod +x "/usr/local/bin/cognito"
        log_success "System-wide launcher created at /usr/local/bin/cognito"
    else
        log_info "Creating user launcher..."
        mkdir -p "$HOME/.local/bin"
        ln -sf "$INSTALL_DIR/cognito" "$HOME/.local/bin/cognito"
        
        # Add to PATH if not already there
        if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc" 2>/dev/null || true
            log_info "Added $HOME/.local/bin to PATH. Please restart your shell or run:"
            log_info "export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
    fi
    
    log_success "Configuration setup complete"
}

# Setup Docker if requested
setup_docker() {
    if [[ "$INSTALL_DOCKER" == true ]] || [[ "$INSTALL_TYPE" == "docker" ]]; then
        log_step "Setting up Docker"
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed. Please install Docker first."
            return 1
        fi
        
        # Build Docker image
        log_info "Building Docker image..."
        cd "$INSTALL_DIR"
        docker build -t cognito:latest .
        docker build -t cognito:$COGNITO_VERSION .
        
        # Create docker-compose file
        cat > "$INSTALL_DIR/docker-compose.yml" << EOF
version: '3.8'

services:
  cognito:
    image: cognito:latest
    ports:
      - "8000:8000"
    environment:
      - COGNITO_ENV=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - cognito
    restart: unless-stopped
EOF
        
        # Create Docker launcher
        cat > "$INSTALL_DIR/cognito-docker" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
docker-compose "\$@"
EOF
        chmod +x "$INSTALL_DIR/cognito-docker"
        
        log_success "Docker setup complete"
    fi
}

# Setup systemd service
setup_systemd_service() {
    if [[ "$SETUP_SYSTEMD" == true ]] && [[ "$OS" == "linux" ]]; then
        log_step "Setting up systemd service"
        
        # Create systemd service file
        cat > "/etc/systemd/system/cognito.service" << EOF
[Unit]
Description=Cognito AI Code Analysis Platform
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/$VENV_NAME/bin
EnvironmentFile=$INSTALL_DIR/config/.env
ExecStart=$INSTALL_DIR/$VENV_NAME/bin/python -m src.main --server
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
        
        # Reload systemd and enable service
        systemctl daemon-reload
        systemctl enable cognito
        
        log_success "Systemd service created. Start with: sudo systemctl start cognito"
    fi
}

# Run post-installation tests
run_tests() {
    log_step "Running post-installation tests"
    
    source "$INSTALL_DIR/$VENV_NAME/bin/activate"
    cd "$INSTALL_DIR"
    
    # Test basic functionality
    log_info "Testing basic functionality..."
    echo 'def hello(): print("world")' | python -m src.main --validate-only
    
    # Test configuration
    log_info "Testing configuration..."
    python -c "from src.config import get_config; print('Config OK')" 2>/dev/null || {
        log_warn "Configuration test failed - this is normal for first install"
    }
    
    # Test language detection
    log_info "Testing language detection..."
    python -c "from src.language_detector import detect_code_language; print('Language detection OK')"
    
    # Run unit tests if available
    if [[ "$INSTALL_TYPE" == "development" ]] && [ -d "tests" ]; then
        log_info "Running unit tests..."
        python -m pytest tests/ -x -q || log_warn "Some tests failed"
    fi
    
    log_success "Basic tests completed"
}

# Setup shell completions
setup_completions() {
    log_step "Setting up shell completions"
    
    # Bash completion
    if command -v bash &> /dev/null; then
        mkdir -p "$HOME/.bash_completion.d"
        cat > "$HOME/.bash_completion.d/cognito" << 'EOF'
_cognito_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="--file --language --output --use-llm --adaptive --report --languages --batch --help --version"
    
    case "${prev}" in
        --language)
            COMPREPLY=($(compgen -W "python c cpp javascript java go rust php ruby csharp" -- ${cur}))
            return 0
            ;;
        --file|--output)
            COMPREPLY=($(compgen -f -- ${cur}))
            return 0
            ;;
        *)
            ;;
    esac
    
    COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
    return 0
}
complete -F _cognito_completion cognito
EOF
        log_info "Bash completion installed"
    fi
    
    # Zsh completion
    if command -v zsh &> /dev/null; then
        mkdir -p "$HOME/.zsh/completions"
        cat > "$HOME/.zsh/completions/_cognito" << 'EOF'
#compdef cognito

_cognito() {
    local context state line
    _arguments \
        '--file[Path to code file]:file:_files' \
        '--language[Programming language]:language:(python c cpp javascript java go rust php ruby csharp)' \
        '--output[Output file]:output:_files' \
        '--use-llm[Use LLM enhancement]' \
        '--adaptive[Use adaptive AI mode]' \
        '--report[Generate improvement report]' \
        '--languages[Show supported languages]' \
        '--batch[Analyze directory]:directory:_directories' \
        '--help[Show help]' \
        '--version[Show version]'
}

_cognito "$@"
EOF
        log_info "Zsh completion installed"
    fi
}

# Display final information
show_installation_summary() {
    log_step "Installation Summary"
    
    echo -e "${GREEN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                  🎉 INSTALLATION COMPLETE! 🎉                ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    echo "Cognito has been successfully installed!"
    echo ""
    echo -e "${CYAN}Installation Details:${NC}"
    echo "  📁 Installation directory: $INSTALL_DIR"
    echo "  🐍 Python environment: $INSTALL_DIR/$VENV_NAME"
    echo "  ⚙️  Configuration: $INSTALL_DIR/config/.env"
    echo "  📊 Data directory: $INSTALL_DIR/data"
    echo "  📝 Logs directory: $INSTALL_DIR/logs"
    echo ""
    
    echo -e "${CYAN}Quick Start:${NC}"
    if command -v cognito &> /dev/null; then
        echo "  🚀 Run: cognito --help"
        echo "  📊 Analyze code: cognito --file your_code.py"
        echo "  🧠 With AI: cognito --file your_code.py --use-llm"
    else
        echo "  🚀 Run: $INSTALL_DIR/cognito --help"
        echo "  📊 Analyze code: $INSTALL_DIR/cognito --file your_code.py"
    fi
    echo ""
    
    if [[ "$ENABLE_LLM" == true ]]; then
        echo -e "${YELLOW}⚠️  LLM Configuration Required:${NC}"
        echo "  1. Get OpenAI API key from: https://platform.openai.com/api-keys"
        echo "  2. Edit: $INSTALL_DIR/config/.env"
        echo "  3. Set: OPENAI_API_KEY=your_key_here"
        echo ""
    fi
    
    if [[ "$INSTALL_DOCKER" == true ]] || [[ "$INSTALL_TYPE" == "docker" ]]; then
        echo -e "${CYAN}Docker Commands:${NC}"
        echo "  🐳 Start: cd $INSTALL_DIR && docker-compose up -d"
        echo "  🛑 Stop: cd $INSTALL_DIR && docker-compose down"
        echo ""
    fi
    
    if [[ "$SETUP_SYSTEMD" == true ]]; then
        echo -e "${CYAN}System Service:${NC}"
        echo "  ▶️  Start: sudo systemctl start cognito"
        echo "  ⏹️  Stop: sudo systemctl stop cognito"
        echo "  📊 Status: sudo systemctl status cognito"
        echo ""
    fi
    
    echo -e "${CYAN}Configuration:${NC}"
    echo "  ⚙️  Edit config: $INSTALL_DIR/config/.env"
    echo "  📝 View logs: tail -f $INSTALL_DIR/logs/cognito.log"
    echo "  🔄 Update: cd $INSTALL_DIR && git pull && pip install -r requirements.txt"
    echo ""
    
    echo -e "${CYAN}Support & Documentation:${NC}"
    echo "  📚 Documentation: https://cognito.readthedocs.io"
    echo "  🐛 Issues: https://github.com/yourusername/cognito/issues"
    echo "  💬 Discussions: https://github.com/yourusername/cognito/discussions"
    echo ""
    
    echo -e "${GREEN}Happy coding with Cognito! 🚀${NC}"
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "Installation failed!"
        echo "Cleaning up..."
        if [ -d "$INSTALL_DIR" ] && [ "$INSTALL_DIR" != "$HOME" ]; then
            read -p "Remove installation directory? [y/N]: " cleanup_choice
            if [[ $cleanup_choice =~ ^[Yy]$ ]]; then
                rm -rf "$INSTALL_DIR"
                log_info "Installation directory removed"
            fi
        fi
    fi
}

# Main installation function
main() {
    trap cleanup EXIT
    
    # Check if running as root when not needed
    if [[ $EUID -eq 0 ]] && [[ "$1" != "--allow-root" ]]; then
        log_warn "Running as root is not recommended. Use --allow-root to override."
        exit 1
    fi
    
    show_logo
    
    local steps=(
        "detect_platform"
        "check_requirements" 
        "configure_installation"
        "install_system_deps"
        "setup_environment"
        "install_cognito"
        "setup_configuration"
        "setup_docker"
        "setup_systemd_service"
        "run_tests"
        "setup_completions"
    )
    
    local total_steps=${#steps[@]}
    local current_step=0
    
    for step in "${steps[@]}"; do
        ((current_step++))
        show_progress $current_step $total_steps
        
        if [[ "$step" == "install_system_deps" ]] && [[ "$INSTALL_TYPE" == "minimal" ]]; then
            continue  # Skip system deps for minimal install
        fi
        
        $step
        sleep 0.5  # Brief pause for visual effect
    done
    
    show_installation_summary
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Cognito Installer"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --version, -v       Show installer version"
        echo "  --allow-root        Allow running as root"
        echo "  --uninstall         Uninstall Cognito"
        echo "  --update            Update existing installation"
        echo ""
        echo "Environment variables:"
        echo "  COGNITO_INSTALL_DIR Directory to install Cognito (default: ~/.cognito)"
        echo ""
        exit 0
        ;;
    --version|-v)
        echo "Cognito Installer v1.0.0"
        exit 0
        ;;
    --uninstall)
        echo "Uninstalling Cognito..."
        if [ -d "$INSTALL_DIR" ]; then
            rm -rf "$INSTALL_DIR"
            rm -f "/usr/local/bin/cognito"
            rm -f "$HOME/.local/bin/cognito"
            rm -f "/etc/systemd/system/cognito.service"
            echo "Cognito uninstalled successfully"
        else
            echo "Cognito is not installed"
        fi
        exit 0
        ;;
    --update)
        echo "Updating Cognito..."
        if [ -d "$INSTALL_DIR" ]; then
            cd "$INSTALL_DIR"
            source "$VENV_NAME/bin/activate"
            git pull
            pip install -r requirements.txt
            echo "Cognito updated successfully"
        else
            echo "Cognito is not installed. Run installer first."
        fi
        exit 0
        ;;
esac

# Run main installation
main "$@"