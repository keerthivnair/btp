import sys
import platform
import os

REQUIRED_PYTHON = "3.11.14"

def check_python_version():
    current_version = platform.python_version()
    print(f"Python version: {current_version}")
    if current_version != REQUIRED_PYTHON:
        print(f"ERROR: Expected Python {REQUIRED_PYTHON}, found {current_version}.")
        sys.exit(1)

def check_packages():
    try:
        import flwr
        print(f"Flower version: {flwr.__version__}")
    except ImportError:
        print("ERROR: flwr not found.")
        sys.exit(1)
        
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("ERROR: torch not found.")
        sys.exit(1)
        
    try:
        import numpy
        print(f"NumPy version: {numpy.__version__}")
    except ImportError:
        print("ERROR: numpy not found.")
        sys.exit(1)

def print_system_info():
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    venv_path = os.environ.get('VIRTUAL_ENV', 'Not inside a virtual environment')
    print(f"Virtual Environment Path: {venv_path}")

if __name__ == "__main__":
    print("--- TRACE-FL Environment Check ---")
    check_python_version()
    print_system_info()
    check_packages()
    print("Environment is configured correctly.")
