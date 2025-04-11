# Crear un archivo .gitignore típico para proyectos Python
gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Caches
*.whl
*.egg-info/
.eggs/
.cache/

# Virtual environment
venv/
env/
ENV/
.venv/

# Environment variables
.env

# System files
.DS_Store
Thumbs.db

# Jupyter Notebooks checkpoints
.ipynb_checkpoints/

# Log files
*.log

# PyInstaller
dist/
build/
*.spec

# VSCode settings
.vscode/
"""

# Guardar en archivo
with open("/mnt/data/.gitignore", "w") as f:
    f.write(gitignore_content)


