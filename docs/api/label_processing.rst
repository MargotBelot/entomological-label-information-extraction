label_processing Package
========================

The ``label_processing`` package contains the core image processing functionality for the Entomological Label Information Extraction system.

.. currentmodule:: label_processing

Package Contents
----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   config
   detect_empty_labels
   label_detection
   label_rotation
   ocr_vision
   tensorflow_classifier
   text_recognition
   utils

Modules
-------

Configuration
~~~~~~~~~~~~~

.. automodule:: label_processing.config
   :members:
   :undoc-members:
   :show-inheritance:

Empty Label Detection
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: label_processing.detect_empty_labels
   :members:
   :undoc-members:
   :show-inheritance:

Label Detection
~~~~~~~~~~~~~~~

.. automodule:: label_processing.label_detection
   :members:
   :undoc-members:
   :show-inheritance:

Label Rotation
~~~~~~~~~~~~~~

.. automodule:: label_processing.label_rotation
   :members:
   :undoc-members:
   :show-inheritance:

OCR Vision
~~~~~~~~~~

.. automodule:: label_processing.ocr_vision
   :members:
   :undoc-members:
   :show-inheritance:

TensorFlow Classifier
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: label_processing.tensorflow_classifier
   :members:
   :undoc-members:
   :show-inheritance:

Text Recognition
~~~~~~~~~~~~~~~~

.. automodule:: label_processing.text_recognition
   :members:
   :undoc-members:
   :show-inheritance:

OCR preprocessing summary
~~~~~~~~~~~~~~~~~~~~~~~~~
The ``text_recognition.ImageProcessor`` applies, prior to Tesseract OCR:
- grayscale conversion
- Gaussian/median denoising
- binarization via Otsu or adaptive mean/Gaussian (block size/C configurable)
- skew estimation within ±10° and deskewing
- optional morphological cleaning (dilation/erosion)

Google Vision OCR is invoked on the rotated ROI without thresholding; word-level bounding boxes are captured via ``ocr_vision``.

Utilities
~~~~~~~~~

.. automodule:: label_processing.utils
   :members:
   :undoc-members:
   :show-inheritance: