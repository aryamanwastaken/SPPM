from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stock-propensity-model",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine learning model to predict stock purchase propensity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-propensity-model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyspark>=3.3.0",
        "xgboost>=1.6.0",
        "scikit-learn>=1.1.0",
        "pandas>=1.4.0",
        "numpy>=1.22.0",
        "boto3>=1.24.0",
        "mlflow>=2.0.0",
        "matplotlib>=3.5.0",
        "plotly>=5.10.0",
        "fastapi>=0.85.0",
        "uvicorn>=0.18.0",
        "python-dotenv>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-propensity-train=stock_propensity_model.main:main",
            "stock-propensity-api=stock_propensity_model.api:main",
            "stock-propensity-monitor=stock_propensity_model.monitoring:main",
        ],
    },
    include_package_data=True,
    package_data={
        "stock_propensity_model": ["config/*.json"],
    },
)