from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trading_backtest",
    version="0.1.0",
    author="Bill Chan",
    author_email="billpwchan@gmail.com",
    description="A professional trading backtest report generator with interactive visualizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/billpwchan/strategy_powerbacktest",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "": ["templates/*.html"],
    },
    install_requires=[
        "futu-api",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "pyyaml>=5.1",
        "typing-extensions>=4.0.0",
        "jinja2>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 