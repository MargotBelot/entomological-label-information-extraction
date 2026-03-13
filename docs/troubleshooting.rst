Troubleshooting
===============

🚑 **Quick Fixes for Common Issues**

.. tip::
   💬 **Still stuck?** Open a `GitHub Issue <https://github.com/MargotBelot/entomological-label-information-extraction/issues>`_ with your error message!

🔧 Installation Issues
-----------------------

.. dropdown:: 🐳 "Docker not found" or "command not found"

   **Problem**: Docker isn't installed or running
   
   **Quick Fix**:
   
   - 🍎 **macOS/Windows**: Download Docker Desktop from https://docker.com and start it
   - 🐧 **Linux**: ``sudo apt install docker.io && sudo systemctl start docker``
   - 🔒 **Linux permissions**: ``sudo usermod -aG docker $USER`` then log out/in

.. dropdown:: 🔑 "GEMINI_API_KEY not set" or "Missing API key"

   **Problem**: The Gemini pipeline requires an API key
   
   **Quick Fix**:
   
   .. code-block:: bash
      
      # Get a free key from https://aistudio.google.com/apikey
      export GEMINI_API_KEY=<your-api-key>
      
      # Or add to your shell profile for persistence:
      echo 'export GEMINI_API_KEY=<your-api-key>' >> ~/.bashrc
      source ~/.bashrc

.. dropdown:: 📦 "ModuleNotFoundError: No module named 'google.genai'"

   **Problem**: The ``google-genai`` package is not installed
   
   **Quick Fix**:
   
   .. code-block:: bash
      
      conda activate ELIE
      pip install google-genai>=1.0.0
      
      # Or reinstall all dependencies:
      pip install -e .

.. dropdown:: 🐍 "Conda environment creation failed"

   **Problem**: Can't create the Python environment
   
   **Quick Fix**:
   
   .. code-block:: bash
      
      # Make sure you're in the right directory
      cd entomological-label-information-extraction
      
      # Update conda and retry
      conda update conda
      conda env create -f environment.yml --force

.. dropdown:: 💻 "Python/pip command not found"

   **Problem**: Python isn't in your PATH
   
   **Quick Fix**:
   
   - Make sure you activated the environment: ``conda activate ELIE``
   - Try ``python3`` instead of ``python``
   - On Windows, use the Anaconda Prompt

🖼️ Processing Issues
------------------------

.. dropdown:: 🔍 "No labels detected" or "Empty results"

   **Problem**: The system isn't finding any labels in your images
   
   **Quick Fix**:
   
   - 📏 Check image quality: Are labels clearly visible?
   - 🔍 Try lowering detection confidence: ``--confidence 0.5``
   - 🖼️ Use single-label pipeline for pre-cropped labels
   - 📊 Check image format (JPEG/PNG work best)

.. dropdown:: 🤖 "OCR results are gibberish"

   **Problem**: Extracted text looks wrong or unreadable
   
   **Quick Fix**:
   
   - 🌐 Try Google Vision instead: ``--ocr-method google`` (more accurate)
   - 🔄 Check if labels need rotation correction
   - 🏷️ Make sure you're using SLI pipeline for individual labels
   - 🗺 Verify language settings match your labels

.. dropdown:: 💻 "Out of memory" or "CUDA errors"

   **Problem**: System runs out of memory during processing
   
   **Quick Fix**:
   
   - 🔋 Process fewer images at once
   - ⚙️ Use smaller batch sizes: ``--batch-size 8``
   - 💾 Close other applications to free memory
   - 🐳 Use Docker (better memory management)

🎁 Quick Help Commands
-------------------------

.. code-block:: bash

   # Check if everything is working
   python scripts/health_check.py
   
   # Test with a single image
python scripts/processing/analysis.py -i path/to/images_dir -o test_output/
   
   # Get help for any script
   python scripts/processing/detection.py --help

📞 Still Need Help?
----------------------

1. 💬 **GitHub Issues**: `Report bugs or ask questions <https://github.com/MargotBelot/entomological-label-information-extraction/issues>`_
2. 📖 **Documentation**: Check the full :doc:`user_guide` for detailed instructions
3. 🤝 **Contributing**: See :doc:`contributing` if you want to help improve the project
4. 🔧 **API Reference**: Check :doc:`api/modules` for technical details

**Low Detection Accuracy**

Symptoms: Few or no labels detected in images

Solutions:
- Check image quality (resolution, lighting, focus)
- Lower confidence threshold: ``--confidence 0.5``
- Verify image format (JPEG, PNG supported)
- Ensure labels are clearly visible in images

**OCR Returns Gibberish**

Symptoms: Extracted text is unreadable or incorrect

Solutions:
- Try Google Vision API instead of Tesseract
- Check image rotation and orientation
- Verify text language settings
- Improve image preprocessing (contrast, denoising)

**Memory Errors**

Error: ``MemoryError`` or ``CUDA out of memory``

Solutions:
- Reduce batch size in processing scripts
- Close other applications to free memory
- Process images sequentially instead of in batches
- Use CPU processing if GPU memory is insufficient

**Slow Processing**

Symptoms: Very long processing times

Solutions:
- Enable GPU acceleration if available
- Reduce image resolution if extremely high
- Process in smaller batches
- Use Docker for optimized resource usage
- Monitor CPU and memory usage

Specific Error Messages
-----------------------

**ImportError: No module named 'cv2'**

Solution:
- Activate the conda environment: ``conda activate ELIE``
- Reinstall OpenCV: ``pip install opencv-python``

**FileNotFoundError: [Errno 2] No such file or directory**

Solution:
- Check file paths are correct and absolute
- Ensure input directories exist and contain images
- Verify output directories are writable

**ModuleNotFoundError: No module named 'tensorflow'**

Solution:
- Check TensorFlow installation: ``pip list | grep tensorflow``
- Reinstall if needed: ``pip install tensorflow>=2.16.0``
- For Apple Silicon Macs: ``pip install tensorflow-macos``

**CUDA Error: device not found**

Solution:
- Install NVIDIA drivers and CUDA toolkit
- Verify GPU compatibility
- Use CPU-only versions if no compatible GPU

Quality Issues
--------------

**Poor Classification Results**

Symptoms: Labels misclassified (empty vs handwritten vs printed)

Solutions:
- Review training data quality
- Adjust classification thresholds
- Manual review of borderline cases
- Retrain models with domain-specific data

**Inconsistent OCR Results**

Symptoms: Same text extracted differently across runs

Solutions:
- Use deterministic processing settings
- Ensure consistent image preprocessing
- Document processing parameters used
- Consider ensemble methods for critical text

**Missing Labels in Detection**

Symptoms: Some labels not detected in multi-label images

Solutions:
- Lower detection confidence threshold
- Check for label overlap or occlusion
- Verify image resolution is sufficient
- Manual annotation of missed cases

Performance Optimization
------------------------

**Speed Improvements**

- Use SSD storage for image directories
- Increase available RAM
- Enable GPU processing for compatible operations
- Process images in parallel where possible
- Optimize image sizes for processing pipeline

**Resource Management**

- Monitor system resources during processing
- Set memory limits for Docker containers
- Use batch processing for large datasets
- Clean up temporary files regularly

**Quality vs Speed Trade-offs**

- Higher confidence thresholds = faster but less complete
- Lower resolution = faster but less accurate
- Tesseract = faster, Google Vision = more accurate
- Batch size affects memory usage and speed

Development Issues
------------------

**Code Changes Not Reflected**

Solution:
- Reinstall package in development mode: ``pip install -e .``
- Restart Python interpreter
- Clear Python cache: ``find . -name "*.pyc" -delete``

**Testing Failures**

Solutions:
- Ensure test data is available
- Check all dependencies are installed
- Run individual test modules to isolate issues
- Update test expectations if API changed

**Documentation Build Fails**

Solutions:
- Install documentation dependencies: ``pip install -e .[docs]``
- Clear Sphinx build cache: ``make clean`` in docs directory
- Check for syntax errors in RST files
- Verify all module imports work correctly

Getting More Help
-----------------

**Log File Analysis**

Important log locations:
- Application logs: ``logs/`` directory
- System logs: ``/var/log/`` (Linux), Console app (macOS)
- Docker logs: ``docker logs <container_name>``

**Diagnostic Information**

When reporting issues, include:
- Operating system and version
- Python version and environment details
- Complete error messages and stack traces
- Input data characteristics (size, format, content)
- Processing parameters used
- System resource availability (RAM, GPU)

**Community Resources**

- GitHub Issues: Report bugs and feature requests
- Documentation: Check latest version online
- Stack Overflow: Search for similar problems
- Research Papers: Understanding algorithm limitations

**Professional Support**

For enterprise users:
- Contact maintainers for dedicated support
- Consider consulting services for custom implementations
- Training workshops available for teams
- Priority bug fixes and feature development

Prevention Tips
---------------

**Regular Maintenance**

- Keep dependencies updated
- Monitor disk space usage
- Regular backup of configuration files
- Test with sample data before large processing runs

**Best Practices**

- Document processing workflows and parameters
- Version control for custom configurations
- Regular validation of output quality
- Maintain separate environments for development and production

**Monitoring**

- Set up automated health checks
- Monitor processing success rates
- Track performance metrics over time
- Alert systems for critical failures