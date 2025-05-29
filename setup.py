from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f.readlines() 
                if line.strip() and not line.startswith('#')]

setup(
    name="cognito",
    version="0.3.0",
    packages=find_packages(where="src"),  
    package_dir={"": "src"},  
    py_modules=["main"],
    python_requires=">=3.8,<3.12",  
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'cognito=main:main',
        ],
    },
)