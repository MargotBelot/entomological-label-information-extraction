#!/usr/bin/env python3
"""
Entomological Label Information Extraction - Desktop GUI
Single-file launcher with complete pipeline control and monitoring.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import subprocess
import threading
import queue
import json
import shutil
from pathlib import Path
import webbrowser

class EntomologicalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Entomological Label Information Extraction")
        self.root.geometry("900x700")
        
        # Get project root
        self.project_root = self.get_project_root()
        
        # Pipeline process tracking
        self.current_process = None
        self.log_queue = queue.Queue()
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.check_system_status()
        
        # Start log monitor
        self.root.after(100, self.update_logs)
    
    def get_project_root(self):
        """Get project root directory"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return script_dir
    
    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        style.theme_use('default')
        
        # Custom colors
        style.configure('Status.TLabel', background='#f0f0f0', padding=5)
        style.configure('Success.TLabel', foreground='#27ae60')
        style.configure('Warning.TLabel', foreground='#f39c12')
        style.configure('Error.TLabel', foreground='#e74c3c')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_pipeline_tab(notebook)
        self.create_scripts_tab(notebook)
        self.create_results_tab(notebook)
        self.create_settings_tab(notebook)
        
        # Status bar at bottom
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready", style='Status.TLabel')
        self.status_label.pack(side='left')
        
        self.help_button = ttk.Button(self.status_frame, text="Help", command=self.show_help)
        self.help_button.pack(side='right')
    
    def create_pipeline_tab(self, notebook):
        """Create main pipeline tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Pipelines")
        
        # Title
        title = ttk.Label(frame, text="AI-Powered Label Processing", style='Title.TLabel')
        title.pack(pady=10)
        
        # System status
        self.system_frame = ttk.LabelFrame(frame, text="System Status", padding=10)
        self.system_frame.pack(fill='x', padx=10, pady=5)
        
        self.docker_status = ttk.Label(self.system_frame, text="Checking Docker...")
        self.docker_status.pack(anchor='w')
        
        self.project_status = ttk.Label(self.system_frame, text="Checking project...")
        self.project_status.pack(anchor='w')
        
        # Pipeline selection
        pipeline_frame = ttk.LabelFrame(frame, text="Select Pipeline", padding=10)
        pipeline_frame.pack(fill='x', padx=10, pady=5)
        
        self.pipeline_var = tk.StringVar(value="mli")
        
        mli_radio = ttk.Radiobutton(pipeline_frame, text="MLI - Multi-Label (Full specimen photos)", 
                                   variable=self.pipeline_var, value="mli")
        mli_radio.pack(anchor='w', pady=2)
        
        sli_radio = ttk.Radiobutton(pipeline_frame, text="SLI - Single-Label (Pre-cropped labels)", 
                                   variable=self.pipeline_var, value="sli")
        sli_radio.pack(anchor='w', pady=2)
        
        # Input/Output paths
        paths_frame = ttk.LabelFrame(frame, text="Data Directories", padding=10)
        paths_frame.pack(fill='x', padx=10, pady=5)
        
        # Input path
        input_frame = ttk.Frame(paths_frame)
        input_frame.pack(fill='x', pady=2)
        ttk.Label(input_frame, text="Input:").pack(side='left')
        self.input_path = tk.StringVar(value=str(Path(self.project_root) / "data" / "MLI" / "input"))
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side='right')
        
        # Output path  
        output_frame = ttk.Frame(paths_frame)
        output_frame.pack(fill='x', pady=2)
        ttk.Label(output_frame, text="Output:").pack(side='left')
        self.output_path = tk.StringVar(value=str(Path(self.project_root) / "data" / "MLI" / "output"))
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side='right')
        
        # Control buttons
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.run_button = ttk.Button(control_frame, text="Start Processing", command=self.run_pipeline, style='Accent.TButton')
        self.run_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_pipeline, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Open Input Folder", command=self.open_input_folder).pack(side='right', padx=5)
        ttk.Button(control_frame, text="Open Output Folder", command=self.open_output_folder).pack(side='right', padx=5)
        
        # Progress
        progress_frame = ttk.LabelFrame(frame, text="Progress", padding=10)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready to start")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.pack(anchor='w')
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)
        
        # Log output
        log_frame = ttk.LabelFrame(frame, text="Processing Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, font=('Courier', 9))
        self.log_text.pack(fill='both', expand=True)
    
    def create_scripts_tab(self, notebook):
        """Create individual scripts tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Individual Scripts")
        
        ttk.Label(frame, text="Individual Processing Scripts", style='Title.TLabel').pack(pady=10)
        
        # Script selection
        script_frame = ttk.LabelFrame(frame, text="Select Script", padding=10)
        script_frame.pack(fill='x', padx=10, pady=5)
        
        self.script_var = tk.StringVar()
        scripts = [
            ("detection", "Label Detection - Find labels in specimen images"),
            ("classifiers", "Classification - Categorize detected labels"),
            ("analysis", "Empty Label Analysis - Quality control"),
            ("tesseract", "OCR Processing - Extract text from labels"),
            ("postprocessing", "Post-processing - Clean and structure text")
        ]
        
        for value, text in scripts:
            ttk.Radiobutton(script_frame, text=text, variable=self.script_var, value=value).pack(anchor='w', pady=2)
        
        # Parameters
        params_frame = ttk.LabelFrame(frame, text="Parameters (Optional)", padding=10)
        params_frame.pack(fill='x', padx=10, pady=5)
        
        self.script_params = tk.StringVar()
        ttk.Entry(params_frame, textvariable=self.script_params, width=70).pack(fill='x')
        ttk.Label(params_frame, text="Leave empty for defaults. Example: -j data/MLI/input -o data/MLI/output").pack(anchor='w', pady=2)
        
        # Script controls
        script_control_frame = ttk.Frame(frame)
        script_control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(script_control_frame, text="Run Script", command=self.run_script).pack(side='left', padx=5)
        ttk.Button(script_control_frame, text="Clear Log", command=self.clear_log).pack(side='left', padx=5)
        
        # Script log
        script_log_frame = ttk.LabelFrame(frame, text="Script Output", padding=10)
        script_log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.script_log = scrolledtext.ScrolledText(script_log_frame, height=15, font=('Courier', 9))
        self.script_log.pack(fill='both', expand=True)
    
    def create_results_tab(self, notebook):
        """Create results viewing tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Results")
        
        ttk.Label(frame, text="Processing Results", style='Title.TLabel').pack(pady=10)
        
        # Results selection
        results_frame = ttk.LabelFrame(frame, text="Select Results", padding=10)
        results_frame.pack(fill='x', padx=10, pady=5)
        
        self.results_type = tk.StringVar(value="mli")
        ttk.Radiobutton(results_frame, text="MLI Results", variable=self.results_type, value="mli").pack(side='left', padx=10)
        ttk.Radiobutton(results_frame, text="SLI Results", variable=self.results_type, value="sli").pack(side='left', padx=10)
        
        ttk.Button(results_frame, text="Refresh", command=self.refresh_results).pack(side='right', padx=5)
        ttk.Button(results_frame, text="Open Results Folder", command=self.open_results_folder).pack(side='right', padx=5)
        
        # Results tree
        tree_frame = ttk.LabelFrame(frame, text="Result Files", padding=10)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_tree = ttk.Treeview(tree_frame, columns=('Size', 'Modified'), show='tree headings')
        self.results_tree.heading('#0', text='File')
        self.results_tree.heading('Size', text='Size')
        self.results_tree.heading('Modified', text='Modified')
        self.results_tree.column('#0', width=300)
        self.results_tree.column('Size', width=100)
        self.results_tree.column('Modified', width=150)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.results_tree.bind('<Double-1>', self.open_result_file)
        
        # Results preview
        preview_frame = ttk.LabelFrame(frame, text="File Preview", padding=10)
        preview_frame.pack(fill='x', padx=10, pady=5)
        
        self.results_preview = scrolledtext.ScrolledText(preview_frame, height=8, font=('Courier', 9))
        self.results_preview.pack(fill='both', expand=True)
    
    def create_settings_tab(self, notebook):
        """Create settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Settings")
        
        ttk.Label(frame, text="System Settings", style='Title.TLabel').pack(pady=10)
        
        # Docker settings
        docker_frame = ttk.LabelFrame(frame, text="Docker Configuration", padding=10)
        docker_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(docker_frame, text="Check Docker Status", command=self.check_docker_status).pack(pady=5)
        ttk.Button(docker_frame, text="Start Docker", command=self.start_docker).pack(pady=5)
        
        # System info
        info_frame = ttk.LabelFrame(frame, text="System Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.system_info = scrolledtext.ScrolledText(info_frame, height=10, font=('Courier', 9))
        self.system_info.pack(fill='both', expand=True)
        
        ttk.Button(info_frame, text="Run Health Check", command=self.run_health_check).pack(pady=5)
        
        # About
        about_frame = ttk.LabelFrame(frame, text="About", padding=10)
        about_frame.pack(fill='x', padx=10, pady=5)
        
        about_text = """Entomological Label Information Extraction v1.0
AI-powered system for museum specimen digitization
Uses computer vision and OCR to extract text from specimen labels

Project: github.com/MargotBelot/entomological-label-information-extraction
License: MIT"""
        
        ttk.Label(about_frame, text=about_text, justify='left').pack(anchor='w')
    
    def check_system_status(self):
        """Check Docker and project status"""
        def check():
            # Check Docker
            try:
                result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.docker_status.config(text="Docker: Available", style='Success.TLabel')
                else:
                    self.docker_status.config(text="Docker: Not found", style='Error.TLabel')
            except:
                self.docker_status.config(text="Docker: Not found", style='Error.TLabel')
            
            # Check project structure
            required_paths = [
                Path(self.project_root) / "data",
                Path(self.project_root) / "tools" / "run_mli_pipeline.sh",
                Path(self.project_root) / "tools" / "run_sli_pipeline.sh"
            ]
            
            if all(p.exists() for p in required_paths):
                self.project_status.config(text="Project: Ready", style='Success.TLabel')
            else:
                self.project_status.config(text="Project: Missing components", style='Warning.TLabel')
        
        threading.Thread(target=check, daemon=True).start()
    
    def browse_input(self):
        """Browse for input directory"""
        folder = filedialog.askdirectory(initialdir=self.input_path.get())
        if folder:
            self.input_path.set(folder)
    
    def browse_output(self):
        """Browse for output directory"""
        folder = filedialog.askdirectory(initialdir=self.output_path.get())
        if folder:
            self.output_path.set(folder)
    
    def open_input_folder(self):
        """Open input folder in file manager"""
        self.open_folder(self.input_path.get())
    
    def open_output_folder(self):
        """Open output folder in file manager"""
        self.open_folder(self.output_path.get())
    
    def open_folder(self, path):
        """Open folder in system file manager"""
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except:
            messagebox.showerror("Error", f"Could not open folder: {path}")
    
    def run_pipeline(self):
        """Run the selected pipeline"""
        if self.current_process:
            messagebox.showwarning("Warning", "A pipeline is already running!")
            return
        
        pipeline_type = self.pipeline_var.get()
        
        # Update paths based on pipeline type
        if pipeline_type == "mli":
            input_dir = Path(self.project_root) / "data" / "MLI" / "input"
            output_dir = Path(self.project_root) / "data" / "MLI" / "output" 
            script_path = Path(self.project_root) / "tools" / "run_mli_pipeline.sh"
        else:
            input_dir = Path(self.project_root) / "data" / "SLI" / "input"
            output_dir = Path(self.project_root) / "data" / "SLI" / "output"
            script_path = Path(self.project_root) / "tools" / "run_sli_pipeline.sh"
        
        # Check input directory
        if not input_dir.exists() or not any(input_dir.glob("*.jpg")) and not any(input_dir.glob("*.png")):
            messagebox.showerror("Error", f"No images found in {input_dir}")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start pipeline
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()
        self.progress_var.set(f"Starting {pipeline_type.upper()} pipeline...")
        
        def run_process():
            try:
                self.current_process = subprocess.Popen(
                    [str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=self.project_root
                )
                
                # Read output line by line
                for line in self.current_process.stdout:
                    self.log_queue.put(('log', line.strip()))
                
                self.current_process.wait()
                
                if self.current_process.returncode == 0:
                    self.log_queue.put(('status', 'Pipeline completed successfully!'))
                else:
                    self.log_queue.put(('status', f'Pipeline failed with code {self.current_process.returncode}'))
                    
            except Exception as e:
                self.log_queue.put(('status', f'Error: {str(e)}'))
            finally:
                self.log_queue.put(('done', None))
                self.current_process = None
        
        threading.Thread(target=run_process, daemon=True).start()
    
    def stop_pipeline(self):
        """Stop the current pipeline"""
        if self.current_process:
            self.current_process.terminate()
            self.log_queue.put(('status', 'Pipeline stopped by user'))
    
    def run_script(self):
        """Run individual script"""
        script = self.script_var.get()
        params = self.script_params.get()
        
        if not script:
            messagebox.showwarning("Warning", "Please select a script")
            return
        
        script_map = {
            'detection': 'src/detection/run_detectron_on_dataset.py',
            'classifiers': 'src/classification/apply_classifier.py', 
            'analysis': 'src/utils/analyse_empty_labels.py',
            'tesseract': 'src/ocr/run_tesseract.py',
            'postprocessing': 'src/postprocessing/postprocess_ocr.py'
        }
        
        script_path = Path(self.project_root) / script_map.get(script, '')
        
        if not script_path.exists():
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return
        
        def run():
            try:
                cmd = ['python3', str(script_path)]
                if params:
                    cmd.extend(params.split())
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                
                self.script_log.delete(1.0, tk.END)
                self.script_log.insert(tk.END, f"Running: {' '.join(cmd)}\n")
                self.script_log.insert(tk.END, f"Exit code: {result.returncode}\n\n")
                self.script_log.insert(tk.END, "STDOUT:\n")
                self.script_log.insert(tk.END, result.stdout)
                self.script_log.insert(tk.END, "\nSTDERR:\n")
                self.script_log.insert(tk.END, result.stderr)
                
            except Exception as e:
                self.script_log.delete(1.0, tk.END)
                self.script_log.insert(tk.END, f"Error: {str(e)}")
        
        threading.Thread(target=run, daemon=True).start()
    
    def clear_log(self):
        """Clear the script log"""
        self.script_log.delete(1.0, tk.END)
    
    def refresh_results(self):
        """Refresh results tree"""
        results_type = self.results_type.get()
        results_dir = Path(self.project_root) / "data" / results_type.upper() / "output"
        
        # Clear tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if not results_dir.exists():
            return
        
        # Populate tree
        for file_path in results_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(results_dir)
                size = file_path.stat().st_size
                modified = file_path.stat().st_mtime
                
                size_str = self.format_size(size)
                modified_str = self.format_time(modified)
                
                self.results_tree.insert("", "end", text=str(rel_path), 
                                       values=(size_str, modified_str),
                                       tags=(str(file_path),))
    
    def format_size(self, size):
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def format_time(self, timestamp):
        """Format timestamp"""
        import datetime
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    
    def open_result_file(self, event):
        """Open selected result file"""
        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            if item['tags']:
                file_path = item['tags'][0]
                self.preview_file(file_path)
    
    def preview_file(self, file_path):
        """Preview file content"""
        try:
            path = Path(file_path)
            if path.suffix.lower() in ['.json', '.csv', '.txt', '.log']:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(5000)  # First 5000 chars
                self.results_preview.delete(1.0, tk.END)
                self.results_preview.insert(1.0, content)
                if len(content) == 5000:
                    self.results_preview.insert(tk.END, "\n\n[File truncated...]")
            else:
                self.results_preview.delete(1.0, tk.END) 
                self.results_preview.insert(1.0, f"Binary file: {path.name}")
        except Exception as e:
            self.results_preview.delete(1.0, tk.END)
            self.results_preview.insert(1.0, f"Error reading file: {e}")
    
    def open_results_folder(self):
        """Open results folder"""
        results_type = self.results_type.get()
        results_dir = Path(self.project_root) / "data" / results_type.upper() / "output"
        if results_dir.exists():
            self.open_folder(str(results_dir))
    
    def check_docker_status(self):
        """Check Docker status"""
        def check():
            try:
                result = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    info = "Docker is running and accessible\n\n" + result.stdout
                else:
                    info = "Docker is installed but not running\n\n" + result.stderr
            except subprocess.TimeoutExpired:
                info = "Docker check timed out - may not be running"
            except FileNotFoundError:
                info = "Docker is not installed or not in PATH"
            except Exception as e:
                info = f"Error checking Docker: {e}"
            
            self.system_info.delete(1.0, tk.END)
            self.system_info.insert(1.0, info)
        
        threading.Thread(target=check, daemon=True).start()
    
    def start_docker(self):
        """Try to start Docker"""
        if sys.platform == "darwin":
            try:
                subprocess.run(['open', '/Applications/Docker.app'], check=True)
                messagebox.showinfo("Info", "Attempting to start Docker Desktop")
            except:
                messagebox.showerror("Error", "Could not start Docker Desktop")
        else:
            messagebox.showinfo("Info", "Please start Docker manually for your platform")
    
    def run_health_check(self):
        """Run system health check"""
        def check():
            try:
                health_script = Path(self.project_root) / "scripts" / "health_check.py"
                if health_script.exists():
                    result = subprocess.run(['python3', str(health_script)], 
                                          capture_output=True, text=True, cwd=self.project_root)
                    output = result.stdout + result.stderr
                else:
                    output = "Health check script not found"
            except Exception as e:
                output = f"Error running health check: {e}"
            
            self.system_info.delete(1.0, tk.END)
            self.system_info.insert(1.0, output)
        
        threading.Thread(target=check, daemon=True).start()
    
    def show_help(self):
        """Show help dialog"""
        help_text = """Entomological Label Information Extraction - Help

Quick Start:
1. Put your images in data/MLI/input (specimen photos) or data/SLI/input (label crops)
2. Select the appropriate pipeline (MLI or SLI)  
3. Click 'Start Processing'
4. Monitor progress in the log window
5. View results in the Results tab

Pipeline Types:
• MLI (Multi-Label): Processes full specimen photos, detects and extracts all labels
• SLI (Single-Label): Processes pre-cropped label images

Individual Scripts:
Use the Scripts tab to run individual processing steps with custom parameters

System Requirements:
• Docker installed and running
• Python 3.10+ recommended
• 5GB+ free disk space

For detailed documentation, see docs/ folder or visit:
github.com/MargotBelot/entomological-label-information-extraction
"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x500")
        
        text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=('Arial', 10))
        text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert(1.0, help_text)
        text.config(state='disabled')
    
    def update_logs(self):
        """Update log display from queue"""
        try:
            while True:
                msg_type, msg = self.log_queue.get_nowait()
                
                if msg_type == 'log':
                    self.log_text.insert(tk.END, msg + '\n')
                    self.log_text.see(tk.END)
                elif msg_type == 'status':
                    self.progress_var.set(msg)
                elif msg_type == 'done':
                    self.run_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    self.progress_bar.stop()
                    self.progress_bar.config(mode='determinate', value=100)
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self.update_logs)

def main():
    """Main function"""
    try:
        root = tk.Tk()
        app = EntomologicalGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()