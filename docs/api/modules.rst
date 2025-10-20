API Reference
=============

Complete technical documentation for developers and advanced users.

.. note::
   👨‍💻 **For most users**: The :doc:`../user_guide` covers everything you need without diving into code details.

Quick Module Guide
------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🖼️ label_processing
      :link: label_processing
      :link-type: doc

      **Core Processing**
      ^^^^^^^^^^^^^^^^^^^
      Detection, classification, OCR, and image processing functions.

   .. grid-item-card:: 🧹 label_postprocessing
      :link: label_postprocessing
      :link-type: doc

      **Text Cleaning**
      ^^^^^^^^^^^^^^^^^
      Post-process and structure OCR results for better quality.

   .. grid-item-card:: 📊 label_evaluation
      :link: label_evaluation
      :link-type: doc

      **Quality Metrics**
      ^^^^^^^^^^^^^^^^^^^
      Evaluate system performance and calculate accuracy metrics.

   .. grid-item-card:: 🐳 pipelines
      :link: pipelines
      :link-type: doc

      **Docker Workflows**
      ^^^^^^^^^^^^^^^^^^^^
      Containerized processing pipelines and configurations.

Detailed Documentation
----------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   label_processing
   label_postprocessing
   label_evaluation
   pipelines
   scripts

Programmatic Usage
------------------

Here's how to use the main components in your Python code:

.. code-block:: python

   # Basic usage example
   from label_processing import LabelDetector, LabelClassifier
   from label_processing.ocr_vision import extract_text
   
   # Detect labels in an image
   detector = LabelDetector()
   bboxes = detector.detect('specimen_photo.jpg')
   
   # Classify cropped labels
   classifier = LabelClassifier()
   label_type = classifier.classify('cropped_label.jpg')
   
   # Extract text with OCR
   text = extract_text('printed_label.jpg', method='tesseract')
