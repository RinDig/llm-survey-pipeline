"""Script to fix all import issues for deployment"""
import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace llm_survey_pipeline imports with correct paths
    replacements = [
        # Config imports
        (r'from llm_survey_pipeline\.config import (.*)', r'from backend.config import \1'),
        (r'from llm_survey_pipeline\.config\.(.*) import', r'from backend.config.\1 import'),
        
        # Core imports  
        (r'from llm_survey_pipeline\.core import (.*)', r'from backend.core import \1'),
        (r'from llm_survey_pipeline\.core\.(.*) import', r'from backend.core.\1 import'),
        
        # Utils imports
        (r'from llm_survey_pipeline\.utils import (.*)', r'from backend.utils import \1'),
        (r'from llm_survey_pipeline\.utils\.(.*) import', r'from backend.utils.\1 import'),
        
        # Models imports
        (r'from llm_survey_pipeline\.models import (.*)', r'from backend.models import \1'),
        (r'from llm_survey_pipeline\.models\.(.*) import', r'from backend.models.\1 import'),
        
        # Main imports
        (r'from llm_survey_pipeline\.main import', r'from main import'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Special case for files in backend directories - use relative imports
    if 'backend' in str(file_path):
        # For files in backend, use relative imports
        backend_replacements = [
            (r'from backend\.config import', r'from config import'),
            (r'from backend\.config\.(.*) import', r'from config.\1 import'),
            (r'from backend\.core import', r'from core import'),
            (r'from backend\.core\.(.*) import', r'from core.\1 import'),
            (r'from backend\.utils import', r'from utils import'),
            (r'from backend\.utils\.(.*) import', r'from utils.\1 import'),
            (r'from backend\.models import', r'from models import'),
            (r'from backend\.models\.(.*) import', r'from models.\1 import'),
        ]
        
        # Check if we're in a subdirectory of backend
        parts = Path(file_path).parts
        if 'backend' in parts:
            backend_index = parts.index('backend')
            depth = len(parts) - backend_index - 2  # -2 for backend folder and filename
            
            if depth == 1:  # In a direct subdirectory of backend
                for pattern, replacement in backend_replacements:
                    # Use .. to go up one level
                    replacement = replacement.replace('from ', 'from ..', 1)
                    content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in: {file_path}")
        return True
    return False

def main():
    """Fix all imports in the project"""
    root_dir = Path(__file__).parent
    
    # Files to fix
    python_files = []
    for pattern in ['**/*.py']:
        python_files.extend(root_dir.glob(pattern))
    
    fixed_count = 0
    for file_path in python_files:
        # Skip this script and __pycache__
        if file_path.name == 'fix_imports.py' or '__pycache__' in str(file_path):
            continue
            
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()