Contributing
============

Thank you for your interest in contributing to the Entomological Label Information Extraction project! This guide will help you get started.

How to Contribute
-----------------

There are many ways to contribute to this project:

- Report bugs and request features
- Improve documentation
- Submit code improvements
- Add new features
- Improve test coverage
- Optimize performance

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork the repository**

   Fork the project on GitHub and clone your fork:

   .. code-block:: bash

      git clone https://github.com/your-username/entomological-label-information-extraction.git
      cd entomological-label-information-extraction

2. **Set up development environment**

   .. code-block:: bash

      # Create conda environment
      conda env create -f environment.yml
      conda activate entomological-label

      # Install in development mode
      pip install -e .[dev]

      # Install pre-commit hooks
      pre-commit install

3. **Create a new branch**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Code Standards
--------------

Style Guidelines
~~~~~~~~~~~~~~~~

- **Python**: Follow PEP 8 style guide
- **Line length**: 88 characters (Black formatter)
- **Imports**: Use isort for import organization
- **Type hints**: Use type annotations where appropriate
- **Docstrings**: Follow Google docstring format

Code Formatting
~~~~~~~~~~~~~~~

The project uses automated code formatting:

.. code-block:: bash

   # Format code with Black
   black .

   # Sort imports with isort
   isort .

   # Run all pre-commit hooks
   pre-commit run --all-files

Testing
-------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_detection.py

   # Run with coverage
   pytest --cov=label_processing

Writing Tests
~~~~~~~~~~~~~

- Write unit tests for all new functions
- Include integration tests for complex workflows
- Test edge cases and error conditions
- Use meaningful test names and docstrings

.. code-block:: python

   def test_detect_labels_with_high_confidence():
       """Test label detection with high confidence threshold."""
       # Test implementation

Documentation
-------------

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

- Use reStructuredText (RST) format
- Include docstrings for all public functions
- Add examples to docstrings
- Update API documentation for new features

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build documentation locally
   cd docs/
   make html

   # View documentation
   open _build/html/index.html

Submitting Changes
------------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. **Ensure tests pass**

   .. code-block:: bash

      pytest
      flake8 .
      mypy label_processing/

2. **Update documentation**

   - Update relevant RST files
   - Add docstrings to new functions
   - Update CHANGELOG.md

3. **Create pull request**

   - Use descriptive title and description
   - Reference related issues
   - Include screenshots if applicable

Pull Request Template
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes

   ## Related Issues
   Fixes #123

   ## Changes Made
   - Added new feature X
   - Fixed bug in Y
   - Updated documentation

   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated

Issue Reporting
---------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

- Operating system and version
- Python version
- Package versions (pip list)
- Minimal reproduction example
- Expected vs actual behavior
- Error messages and stack traces

Feature Requests
~~~~~~~~~~~~~~~~

For feature requests, please describe:

- Use case and motivation
- Proposed solution
- Alternative solutions considered
- Impact on existing functionality

Development Guidelines
----------------------

Architecture Principles
~~~~~~~~~~~~~~~~~~~~~~~

- **Modularity**: Keep components loosely coupled
- **Testability**: Write testable code
- **Documentation**: Document public APIs
- **Performance**: Consider computational efficiency
- **Maintainability**: Write clear, readable code

Adding New Features
~~~~~~~~~~~~~~~~~~~

1. **Design phase**
   - Create design document for significant features
   - Get feedback from maintainers
   - Consider backward compatibility

2. **Implementation phase**
   - Follow existing code patterns
   - Add comprehensive tests
   - Update documentation

3. **Review phase**
   - Self-review all changes
   - Address reviewer feedback
   - Ensure CI passes

Model Contributions
-------------------

Contributing Models
~~~~~~~~~~~~~~~~~~~

If contributing new models:

- Include model architecture details
- Provide training data information
- Document model performance metrics
- Include example usage code
- Consider model size and inference speed

Model Standards
~~~~~~~~~~~~~~~

- Use PyTorch or TensorFlow frameworks
- Include model validation code
- Provide model conversion utilities
- Document hardware requirements

Community
---------

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and discussion

Code of Conduct
~~~~~~~~~~~~~~~

This project follows a Code of Conduct. Please be respectful and inclusive in all interactions.

Recognition
-----------

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- Documentation acknowledgments
- Academic publications (where appropriate)

Release Process
---------------

Version Numbering
~~~~~~~~~~~~~~~~~

The project follows Semantic Versioning:

- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

Release Checklist
~~~~~~~~~~~~~~~~~

For maintainers preparing releases:

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test documentation
5. Create release notes
6. Tag release in Git
7. Deploy to package repositories

Getting Help
------------

If you need help with development:

1. Check existing documentation
2. Search GitHub issues
3. Ask questions in GitHub Discussions
4. Contact maintainers directly

Thank you for contributing to the project!