"""
Syda - Synthetic Data Generation Library

A Python library for AI-powered synthetic data generation with referential integrity.
Supports multiple AI providers (OpenAI, Anthropic) and various schema formats.
"""

from .generate import SyntheticDataGenerator
from .schemas import ModelConfig, ProxyConfig

__all__ = [
    'SyntheticDataGenerator',
    'ModelConfig', 
    'ProxyConfig'
]

__version__ = '0.0.2'
__author__ = 'Rama Krishna Kumar Lingamgunta'
__email__ = 'lrkkumar2606@gmail.com'
__license__ = 'LGPL-3.0-or-later'
__description__ = 'A Python library for AI-powered synthetic data generation with referential integrity'
