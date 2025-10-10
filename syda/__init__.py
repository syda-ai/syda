"""
Syda - Synthetic Data Generation Library

A Python library for AI-powered synthetic data generation with referential integrity.
Supports multiple AI providers (OpenAI, Anthropic) and various schema formats.
"""

from .generate import SyntheticDataGenerator
from .schemas import ModelConfig

__all__ = [
    'SyntheticDataGenerator',
    'ModelConfig'
]

__version__ = '0.0.4'
__author__ = 'Rama Krishna Kumar Lingamgunta'
__email__ = 'ramkumar2606@gmail.com'
__license__ = 'MIT'
__description__ = 'Seamlessly generates realistic synthetic test data—including structured, unstructured, PDF, and HTML—using AI and large language models. It preserves referential integrity, maintains privacy compliance, and accelerates development workflows. SYDA enables both highly regulated industries such as healthcare and banking, as well as non-regulated environments like software testing, research, and analytics, to safely simulate diverse data scenarios without exposing sensitive information.'
