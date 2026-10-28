from setuptools import setup, find_packages

setup(
    name="late-edition",
    version="0.1.0",
    description="Public-domain historic newspaper OCR correction with Gemma 4 and safety guards",
    py_modules=["cli", "fetch", "chunk", "correct", "guard", "report", "config"],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["templates/*.jinja2"],
    },
    install_requires=[
        "click==8.1.8",
        "google-genai==2.31.0",
        "jinja2==3.1.6",
        "pytest==9.1.1",
        "python-dotenv==1.2.4",
        "requests==2.32.3",
        "typer==0.15.1",
    ],
    entry_points={
        "console_scripts": [
            "late-edition=cli:main",
        ],
    },
    python_requires=">=3.11",
)
