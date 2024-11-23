from setuptools import setup, find_packages

setup(
    name="trading_backtest",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "futu-api",
        "pandas",
        "numpy",
        "pyyaml",
        "typing-extensions",
    ],
) 