# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.3] - 2025-09-21

### Added
- Azure OpenAI provider support for enterprise deployments
- Advanced configuration with `extra_kwargs` parameter for all providers
- AI gateway integration support (LiteLLM, Portkey, Kong, and custom gateways)
- Comprehensive Azure OpenAI documentation and examples
- Enhanced model configuration guide with `extra_kwargs` reference
- Support for custom endpoints, authentication headers, and timeouts
- Enterprise-grade features for production deployments

### Changed
- Development status upgraded from Beta to Production/Stable
- Enhanced documentation with AI gateway integration examples
- Improved error handling and troubleshooting guidance
- Updated model configuration documentation with provider-specific examples

### Fixed
- Enhanced provider-specific parameter handling
- Better error messages for configuration issues


## [0.0.2] - 2025-08-23

### Added
- Support for Google Gemini Models

### Changed
- Documentation Fixes


## [0.0.1] - 2025-08-11

### Added
- Modern packaging with pyproject.toml
- Support for multiple AI providers (OpenAI, Anthropic Claude)
- Comprehensive schema formats (SQLAlchemy, YAML, JSON, Dict)
- Foreign key relationship handling with referential integrity
- Unstructured document generation with templates
- Custom generators for domain-specific data
- Multi-provider AI integration with consistent interface
- Automatic dependency resolution via topological sorting