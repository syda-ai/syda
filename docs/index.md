# Syda Documentation

A Python-based open-source library for generating synthetic data with AI while preserving referential integrity. Syda allows seamless use of OpenAI, Anthropic (Claude), and other AI models to create realistic synthetic data.

## What is Syda?

Syda is a powerful library that helps developers and data scientists generate high-quality synthetic data using large language models. Whether you need to create test data for your application, training datasets for machine learning, or mock data that maintains complex relationships between tables, Syda provides an elegant solution.

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
