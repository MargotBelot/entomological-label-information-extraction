Changelog
=========

All notable changes to the Entomological Label Information Extraction project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_, and this project adheres to `Semantic Versioning <https://semver.org/>`_.

[2.0.0] - 2025-03-13
---------------------

Added
~~~~~
- **Gemini Pipeline**: New recommended pipeline using Google Gemini API for detection, classification, OCR, and handwritten text recognition (HTR)
- **Entity Recognition**: Automated extraction of structured entities (scientific names, collectors, dates, localities) from OCR text
- **GBIF Validation**: Scientific name validation against the GBIF Backbone Taxonomy
- **OSM Geocoding**: Locality geocoding via OpenStreetMap Nominatim
- **Darwin Core Export**: Standardised Darwin Core records in JSON and CSV formats
- **Streamlit OCR Correction**: Edit transcribed text directly in the browser and re-run entity recognition
- **Streamlit Entity Viewer**: Browse extracted entities per label with GBIF/OSM enrichment
- **Docker Gemini Profile**: Lightweight ``docker-compose --profile gemini`` for cloud-based processing
- **Gemini Requirements**: Minimal ``pipelines/requirements/gemini.txt`` for the Gemini Docker stage
- **Gemini Pipeline Script**: ``tools/pipelines/run_gemini_pipeline_conda.sh`` for command-line execution
- New CLI scripts: ``scripts/processing/gemini_classify.py``, ``scripts/processing/gemini_ocr.py``, ``scripts/processing/entity_recognition.py``
- New modules: ``label_processing/gemini_processor.py``, ``label_processing/entity_recognition.py``
- Unit tests for ``gemini_processor.py`` and ``entity_recognition.py``
- Comprehensive Sphinx documentation with Read the Docs integration

Changed
~~~~~~~
- Conda environment renamed from ``entomological-label`` to ``ELIE``
- Removed all hardcoded API keys from scripts
- Warning suppressions centralised in ``label_processing/__init__.py``
- Improved Streamlit interface with pipeline selection, progress tracking, and export features
- Updated documentation structure for v2.0
- Version bumped to 2.0.0 in ``pyproject.toml`` and Sphinx ``conf.py``

Fixed
~~~~~
- Fixed ``check_text`` → ``check_nuri_format`` call in ``ocr_vision.py``
- Fixed broken imports in test modules
- Fixed redundant exception handling in ``utils.py``
- Updated stale docstrings in ``text_recognition.py``

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

Version 2.1.0
~~~~~~~~~~~~~~
- Enhanced multi-language support for entity recognition
- Batch Gemini API processing with rate-limit management
- Advanced clustering analysis integration (ELIE-clustering)
- Extended format support (TIFF, WebP)
- RESTful API for remote processing

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