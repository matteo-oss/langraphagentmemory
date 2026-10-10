"""
conftest.py — Pytest Configuration

This file is automatically loaded by pytest before any test collection begins.
Its job is to load environment variables from .env so that all modules
(LLM clients, API keys, etc.) can initialize correctly during import.

Place this file at the project root (same level as tests/).
"""

from dotenv import load_dotenv

# Load .env before any test module is imported.
# This ensures GROQ_API_KEY, GEMINI_API_KEY, API_KEY, etc.
# are available when LLM clients are instantiated at module load time.
load_dotenv()