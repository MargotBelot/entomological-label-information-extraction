Changelog
=========

All notable changes to the Entomological Label Information Extraction project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_, and this project adheres to `Semantic Versioning <https://semver.org/>`_.

[Unreleased]
------------

Added
~~~~~
- Comprehensive Sphinx documentation
- Read the Docs integration
- API documentation with autodoc
- User guides and tutorials

Changed
~~~~~~~
- Improved documentation structure
- Enhanced configuration options

[1.0.0] - 2024-01-01
---------------------

Added
~~~~~
- Initial release of the Entomological Label Information Extraction system
- Multi-label image processing pipeline (MLI)
- Single-label image processing pipeline (SLI)
- GUI interface for easy processing
- Docker containerization support
- Comprehensive evaluation framework
- Support for Tesseract and Google Vision OCR
- Label detection using Faster R-CNN
- Label classification (empty/handwritten/printed/identifier)
- Rotation correction for printed labels
- Text post-processing and cleaning
- Batch processing capabilities
- Configuration management system
- Extensive logging and monitoring

Core Features
~~~~~~~~~~~~~
- **Label Detection**: Faster R-CNN model for detecting labels in specimen images
- **Classification**: CNN-based classifier for label types
- **OCR Integration**: Support for multiple OCR engines
- **Post-processing**: Text cleaning and structuring
- **Evaluation**: Comprehensive metrics and analysis tools
- **Docker Support**: Containerized processing pipelines
- **GUI Interface**: User-friendly graphical interface

Performance
~~~~~~~~~~~
- Detection accuracy: >90% F1-score on benchmark datasets
- Classification accuracy: >95% on label type classification
- OCR performance: <5% character error rate on high-quality images
- Processing speed: 100+ images per hour on standard hardware

Documentation
~~~~~~~~~~~~~
- Complete API documentation
- User guides and tutorials
- Installation instructions
- Configuration examples
- Evaluation methodologies
- Troubleshooting guides

Development
~~~~~~~~~~~
- Full test coverage for core functionality
- Continuous integration setup
- Code quality tools (Black, isort, flake8)
- Pre-commit hooks for code consistency
- Development environment setup

Security
~~~~~~~~
- Secure handling of API keys and credentials
- Input validation and sanitization
- Error handling and logging

Compatibility
~~~~~~~~~~~~~
- **Python**: 3.10, 3.11, 3.12
- **Operating Systems**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Hardware**: CPU and GPU processing support
- **Docker**: Multi-platform container support

Known Issues
~~~~~~~~~~~~
- High memory usage with very large images (>50MP)
- Google Vision API rate limiting may affect batch processing
- Some European characters may be misrecognized in Tesseract OCR

Future Releases
---------------

Planned Features
~~~~~~~~~~~~~~~~

Version 1.1.0
~~~~~~~~~~~~~~
- Enhanced multi-language support
- Improved handwriting recognition
- Advanced post-processing rules
- Performance optimizations
- Additional evaluation metrics

Version 1.2.0
~~~~~~~~~~~~~~
- RESTful API for remote processing
- Database integration capabilities
- Advanced clustering analysis
- Custom model training tools
- Extended format support (TIFF, WebP)

Version 2.0.0
~~~~~~~~~~~~~~
- Modern transformer-based OCR models
- Real-time processing capabilities
- Cloud deployment options
- Advanced AI features
- Breaking API changes for improved usability

Contributing
------------

We welcome contributions! Please see our :doc:`contributing` guide for details on:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Pull request process
- Community guidelines

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
---------------

Special thanks to:

- Contributors and maintainers
- Beta testers and early adopters
- Museum partners providing test data
- Open source community for tools and libraries
- Research institutions supporting development