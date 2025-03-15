from setuptools import setup, find_packages

setup(
    name="cognito",
    version="0.2.0",
    packages=find_packages(),
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
    ],
    entry_points={
        'console_scripts': [
            'cognito=main:main',  
        ],
    },
)