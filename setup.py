from setuptools import setup, find_packages
import os

def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r', encoding='utf-8') as f:
        return [
            line.strip() for line in f.readlines() 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]

def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AI-powered multi-language code analysis platform"

setup(
    name="cognito",
    version="0.8.0",  
    description="AI-powered multi-language code analysis platform",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/Klus3kk/cognito",
    
    # Package configuration
    packages=find_packages(where="src"),  
    package_dir={"": "src"},  
    py_modules=["main"],
    
    # Python version support (updated for modern compatibility)
    python_requires=">=3.8,<3.13",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'tensorflow': ['tensorflow>=2.6.0'],
        'audio': ['torch-audio>=2.0.0'],
        'full': [
            'tensorflow>=2.6.0',
            'torch-audio>=2.0.0',
        ]
    },
    
    # CLI entry point
    entry_points={
        'console_scripts': [
            'cognito=main:main',
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
    
    # Include data files
    include_package_data=True,
    package_data={
        'cognito': ['data/*', 'models/*'],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/Klus3kk/cognito/issues",
        "Source": "https://github.com/Klus3kk/cognito",
        "Documentation": "https://cognito.readthedocs.io/",
    },
)