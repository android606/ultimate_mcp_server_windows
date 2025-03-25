#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages


def read(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def get_version():
    version_file = read("llm_gateway/__init__.py")
    version_match = re.search(r"""^__version__ = ["']([^"']*)["']""", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_requirements():
    """Parse requirements from pyproject.toml"""
    requirements = []
    try:
        with open("pyproject.toml", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Basic parsing of dependencies section
        deps_match = re.search(r"\[project.dependencies\](.*?)(\[|\Z)", content, re.DOTALL)
        if deps_match:
            deps_section = deps_match.group(1)
            for line in deps_section.strip().split("\n"):
                line = line.strip()
                if line and "=" in line:
                    # Extract package name
                    package = line.split("=")[0].strip().strip('"\'')
                    if package:
                        requirements.append(package)
        
        return requirements
    except Exception:
        # Fallback to minimal requirements
        return [
            "mcp>=0.2.0",
            "anthropic>=0.8.0",
            "openai>=1.3.0",
            "google-generativeai>=0.3.0",
            "httpx>=0.24.1",
            "pydantic>=2.0.0",
            "rich>=13.0.0",
            "python-dotenv>=1.0.0",
        ]


setup(
    name="llm_gateway_mcp_server",
    version=get_version(),
    description="A Model Context Protocol (MCP) server for accessing multiple LLM providers with cost optimization",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Jeffrey Emanuel",
    author_email="jeffrey.emanuel@gmail.com",
    url="https://github.com/Dicklesworthstone/llm_gateway_mcp_server",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "ruff",
            "mypy",
        ],
        "docs": [
            "mkdocs",
            "mkdocs-material",
            "mkdocstrings",
        ],
        "vector": [
            "numpy",
            "sentence-transformers",
            "chromadb",
        ],
        "all": [
            "numpy",
            "sentence-transformers",
            "chromadb",
            "transformers",
            "torch",
            "pandas",
            "tiktoken",
            "asyncio",
            "aiofiles",
            "diskcache",
            "jsonschema",
        ],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "llm-gateway=llm_gateway.cli.main:main",
        ],
    },
    keywords=[
        "mcp", 
        "llm", 
        "ai", 
        "gateway", 
        "api", 
        "openai", 
        "anthropic", 
        "claude", 
        "gpt", 
        "gemini",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)