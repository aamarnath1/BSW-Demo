from setuptools import setup, find_packages

setup(
    name="vwap_bot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'backtrader',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'optuna'
    ],
) 