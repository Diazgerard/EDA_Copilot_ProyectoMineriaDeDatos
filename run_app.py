#!/usr/bin/env python
"""
Script para ejecutar la aplicación EDA Copilot con Streamlit
"""
import subprocess
import sys
import os

def main():
    # Cambiar al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Ejecutar streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "eda_copilot/app_streamlit.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
