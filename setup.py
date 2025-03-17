from setuptools import setup, find_packages

setup(
    name="cognito",
    version="0.3.0",
    packages=find_packages(where="src"),  
    package_dir={"": "src"},  
    py_modules=["main"],
    install_requires=[
        "pytest",
        "nltk", 
        "transformers",
        "datasets",
        "huggingface-hub",
        "memory_profiler",
        "torch",
        "colorama",
        "langchain",
        "openai",
    ],
    entry_points={
        'console_scripts': [
            'cognito=src.main:main',  
        ],
    },
)