---
title: Syda - AI-Powered Synthetic Test Data Generation | Python Library
description: Generate realistic synthetic test data with AI - structured, unstructured, PDF, and HTML data generation using OpenAI, Anthropic Claude, and Google Gemini while preserving referential integrity and maintaining privacy compliance.
keywords:
  - synthetic data generation
  - test data generation
  - AI data generation
  - large language models
  - OpenAI GPT
  - Anthropic Claude
  - Google Gemini
  - SQLAlchemy
  - referential integrity
  - privacy compliance
  - structured data
  - unstructured data
  - PDF generation
  - HTML generation
  - python library
---

# What is Syda?

Syda seamlessly generate realistic synthetic test data - structured, unstructured, PDF, and HTML data generation with AI and large language models while preserving referential integrity, maintaining privacy compliance, and accelerating development workflows using OpenAI, Anthropic, and Gemini models

## Key Features

- **Multi-Provider AI Integration**:

    * Seamless integration with multiple AI providers
    * Support for OpenAI (GPT), Anthropic (Claude) and Google(Gemini). 
    * Default model is Anthropic Claude model claude-3-5-haiku-20241022
    * Consistent interface across different providers
    * Provider-specific parameter optimization

- **LLM-based Data Generation**:

    * AI-powered schema understanding and data creation
    * Contextually-aware synthetic records
    * Natural language prompt customization
    * Intelligent schema inference


  
- **Multiple Schema Formats**:
    
    * YAML/JSON schema file support with full foreign key relationship handling
    * SQLAlchemy model integration with automatic metadata extraction
    * Python dictionary-based schema definitions
  
- **Referential Integrity**

    * Automatic foreign key detection and resolution
    * Multi-model dependency analysis through topological sorting
    * Robust handling of related data with referential constraints
  
- **Custom Generators**

    * Register column- or type-specific functions for domain-specific data
    * Contextual generators that adapt to other fields (like ICD-10 codes based on demographics)
