"""
config.example.py
=================
Template configuration for the Moon Survival experiment.
Rename this file to `config.py` in your local clone and fill in values.
"""
import os

# Put your OpenAI API key here, or leave it to use the environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

# Default model to use
DEFAULT_MODEL = "gpt-4o-mini"
