from setuptools import setup, find_packages

setup(
    name="cognito",
    version="0.1.0",
    packages=find_packages(),
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
            'cognito=src.main:main',
        ],
    },
)