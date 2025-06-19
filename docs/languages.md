# Language Support

Cognito provides comprehensive analysis for multiple programming languages with tiered support levels.

## Support Tiers

### Tier 1 - Full Support
Complete analysis with specialized analyzers, security checks, and performance optimization.

### Tier 2 - Core Support  
Universal analysis plus basic language-specific patterns.

### Tier 3 - Generic Support
Pattern-based analysis with generic complexity and maintainability metrics.

## Supported Languages

### Python (Tier 1)
**File Extensions**: `.py`, `.pyw`, `.pyi`  
**Analyzer**: `src/analyzers/python_analyzer.py`

**Analysis Features:**
- **Security**: SQL injection, eval() usage, pickle vulnerabilities, input validation
- **Performance**: List comprehensions, generator usage, algorithmic complexity
- **Style**: PEP 8 compliance, naming conventions, import organization
- **Readability**: Docstring coverage, function complexity, code structure

### C (Tier 1)
**File Extensions**: `.c`, `.h`  
**Analyzer**: `src/analyzers/c_analyzer.py`

**Analysis Features:**
- **Security**: Buffer overflows, format string vulnerabilities, integer overflows
- **Memory**: Memory leaks, uninitialized variables, dangling pointers
- **Performance**: Loop optimization, function call overhead
- **Safety**: Unsafe function usage, bounds checking

### C++ (Tier 1)
**File Extensions**: `.cpp`, `.cxx`, `.cc`, `.hpp`, `.hxx`  
**Analyzer**: `src/analyzers/cpp_analyzer.py`

**Analysis Features:**
- **Modern C++**: RAII usage, smart pointers, move semantics
- **Security**: Same as C plus STL-specific issues
- **Performance**: Template efficiency, object construction/destruction
- **Best Practices**: Exception safety, const correctness

### Java (Tier 1)
**File Extensions**: `.java`  
**Analyzer**: `src/analyzers/java_analyzer.py`

**Analysis Features:**
- **Security**: Serialization vulnerabilities, SQL injection, path traversal
- **Enterprise**: Spring framework patterns, design patterns
- **Performance**: Collection usage, string concatenation, autoboxing
- **Concurrency**: Thread safety, synchronization issues

### JavaScript (Tier 1)
**File Extensions**: `.js`, `.jsx`, `.mjs`  
**Analyzer**: `src/analyzers/javascript_analyzer.py`

**Analysis Features:**
- **Security**: XSS prevention, eval() usage, prototype pollution
- **Modern JS**: ES6+ features, async/await patterns
- **DOM Safety**: Event handling, input sanitization
- **Node.js**: Server-side specific checks, package security

### TypeScript (Tier 2)
**File Extensions**: `.ts`, `.tsx`  
**Analysis**: Universal analyzer + TypeScript patterns

**Features:**
- Type safety analysis
- Generic complexity metrics
- Basic security patterns
- Code structure analysis

### Go (Tier 2)
**File Extensions**: `.go`  
**Analysis**: Universal analyzer + Go patterns

**Features:**
- Goroutine usage patterns
- Error handling analysis
- Basic security checks
- Performance patterns

### Rust (Tier 2)
**File Extensions**: `.rs`  
**Analysis**: Universal analyzer + Rust patterns

**Features:**
- Memory safety (automatic)
- Ownership pattern analysis
- Error handling with Result types
- Performance characteristics

### PHP (Tier 2)
**File Extensions**: `.php`, `.phtml`  
**Analysis**: Universal analyzer + PHP patterns

**Features:**
- Basic SQL injection detection
- Variable usage patterns
- Include/require analysis
- Basic security patterns

### Ruby (Tier 2)
**File Extensions**: `.rb`, `.rake`  
**Analysis**: Universal analyzer + Ruby patterns

**Features:**
- Rails framework patterns
- Method naming conventions
- Block usage analysis
- Basic security patterns

### Other Languages (Tier 3)
**Supported**: Any programming language  
**Analysis**: Generic pattern-based analysis

**Languages Include:**
- Swift, Kotlin, Scala, Haskell
- R, MATLAB, Julia
- Shell scripts (bash, zsh)
- Perl, Lua, Dart
- And many more...

**Features:**
- Complexity analysis
- Basic naming conventions
- Function/method detection
- Comment ratio analysis
- Line length and formatting

## Language Detection

Cognito automatically detects programming languages using multiple strategies:

### Detection Methods
1. **File Extension Mapping** - Primary detection method
2. **Content Analysis** - Pattern-based detection for unknown extensions
3. **Shebang Detection** - For script files without extensions
4. **Keyword Analysis** - Language-specific syntax patterns

### Confidence Scoring
- **90-100%**: High confidence, specialized analyzer used
- **70-89%**: Good confidence, universal analyzer with language hints
- **50-69%**: Moderate confidence, universal analyzer
- **Below 50%**: Generic analysis only
