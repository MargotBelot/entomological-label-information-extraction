#!/usr/bin/env python3
"""
ELIE - Quick Launcher
Simple script to launch the Streamlit interface from the root directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit interface"""
    # Get project root (where this script is located)
    project_root = Path(__file__).parent
    
    # Path to the Streamlit interface
    interface_path = project_root / "interfaces" / "launch_streamlit.py"
    
    if not interface_path.exists():
        print("Streamlit interface not found at:", interface_path)
        print("Please ensure the interfaces directory contains launch_streamlit.py")
        sys.exit(1)
    
    print("Launching ELIE Streamlit Interface...")
    print(f"Interface: {interface_path}")
    print("Press Ctrl+C to stop")
    print()
    
    # Change to project root directory
    os.chdir(project_root)
    
    try:
        # Launch Streamlit with optimized configuration and cache clearing
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(interface_path),
            "--server.address", "localhost", 
            "--browser.gatherUsageStats", "false",
            "--server.enableStaticServing", "false",  # Disable caching
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",  # Disable to avoid CORS conflicts
            "--global.suppressDeprecationWarnings", "true",  # Suppress deprecation warnings
            "--theme.base", "light"
        ])
    except KeyboardInterrupt:
        print("\nELIE interface stopped.")
    except FileNotFoundError:
        print("Streamlit not found. Please install it with:")
        print("pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()