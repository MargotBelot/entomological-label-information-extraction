License
=======

MIT License
-----------

Copyright (c) 2024 Margot Belot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Third-Party Licenses
--------------------

This project incorporates several open-source libraries and tools, each with their own licenses:

Core Dependencies
~~~~~~~~~~~~~~~~~

**TensorFlow**
    Apache License 2.0
    https://github.com/tensorflow/tensorflow/blob/master/LICENSE

**PyTorch** 
    BSD-3-Clause License
    https://github.com/pytorch/pytorch/blob/master/LICENSE

**OpenCV**
    Apache License 2.0
    https://github.com/opencv/opencv/blob/master/LICENSE

**NumPy**
    BSD-3-Clause License
    https://github.com/numpy/numpy/blob/main/LICENSE.txt

**Pandas**
    BSD-3-Clause License
    https://github.com/pandas-dev/pandas/blob/main/LICENSE

**Matplotlib**
    PSF-based License
    https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE

**scikit-learn**
    BSD-3-Clause License
    https://github.com/scikit-learn/scikit-learn/blob/main/COPYING

**Pillow**
    HPND License
    https://github.com/python-pillow/Pillow/blob/main/LICENSE

OCR Dependencies
~~~~~~~~~~~~~~~~

**Tesseract**
    Apache License 2.0
    https://github.com/tesseract-ocr/tesseract/blob/master/LICENSE

**pytesseract**
    Apache License 2.0
    https://github.com/madmaze/pytesseract/blob/master/LICENSE

**Google Cloud Vision**
    Apache License 2.0
    https://cloud.google.com/terms/

Web and GUI Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

**Flask**
    BSD-3-Clause License
    https://github.com/pallets/flask/blob/main/LICENSE.rst

**Flask-CORS**
    MIT License
    https://github.com/corydolphin/flask-cors/blob/master/LICENSE

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

**pytest**
    MIT License
    https://github.com/pytest-dev/pytest/blob/main/LICENSE

**Black**
    MIT License
    https://github.com/psf/black/blob/main/LICENSE

**isort**
    MIT License
    https://github.com/PyCQA/isort/blob/main/LICENSE

**flake8**
    MIT License
    https://github.com/PyCQA/flake8/blob/main/LICENSE

Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sphinx**
    BSD-2-Clause License
    https://github.com/sphinx-doc/sphinx/blob/master/LICENSE

**Read the Docs Sphinx Theme**
    MIT License
    https://github.com/readthedocs/sphinx_rtd_theme/blob/master/LICENSE

Models and Training Data
-------------------------

Detection Models
~~~~~~~~~~~~~~~~

The label detection models included in this project are trained on curated datasets of entomological specimen labels. These models are provided under the same MIT license as the main software.

Classification Models
~~~~~~~~~~~~~~~~~~~~~

The label classification models are trained on publicly available datasets and proprietary museum collections. The models themselves are licensed under MIT, but users should respect the terms of the original training data sources.

Usage Terms
-----------

Commercial Use
~~~~~~~~~~~~~~

This software is free for commercial use under the MIT license terms. However, please note:

- Google Cloud Vision API usage is subject to Google's pricing and terms
- Some training datasets may have restrictions on commercial use
- Users are responsible for compliance with all applicable laws and regulations

Academic Use
~~~~~~~~~~~~

This software is freely available for academic research and educational purposes. If you use this software in academic work, please consider citing:

[Citation information will be provided upon publication]

Attribution
~~~~~~~~~~~

While not required by the MIT license, attribution is appreciated:

- Include a reference to this project in derivative works
- Acknowledge the contributors in academic publications
- Consider contributing improvements back to the project

Disclaimer
----------

This software is provided "as is" without warranty of any kind. The authors make no representations about the suitability of this software for any purpose. Users are responsible for:

- Validating results for their specific use case
- Ensuring compliance with applicable regulations
- Maintaining appropriate data privacy and security
- Testing thoroughly before production use

The performance of machine learning models may vary depending on:

- Input image quality
- Domain-specific characteristics
- Hardware capabilities
- Configuration settings

Contact
-------

For licensing questions or commercial inquiries, please contact:

Margot Belot
Email: [contact information]
GitHub: https://github.com/your-username/entomological-label-information-extraction

Legal Notice
------------

This software may be subject to export control laws and regulations. Users are responsible for compliance with all applicable laws in their jurisdiction.

The use of this software for processing copyrighted or sensitive material is the sole responsibility of the user. Always ensure you have appropriate permissions before processing images or documents.