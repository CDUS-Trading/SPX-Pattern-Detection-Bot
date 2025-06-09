from setuptools import setup, find_packages

setup(
    name="MINUTE-DATA-PATTERN-TRADING",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "pandas_market_calendars",
    ],
    python_requires=">=3.7",
) 