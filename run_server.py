#!/usr/bin/env python
"""
Startup script for the FastAPI server with proper reload exclusions.
This prevents uvicorn from trying to watch node_modules and other unnecessary directories.
"""
import uvicorn
from pathlib import Path

if __name__ == "__main__":
    # Exclude common directories that shouldn't be watched for changes
    reload_excludes = [
        "*/node_modules/*",
        "*/__pycache__/*",
        "*/.git/*",
        "*/venv/*",
        "*/env/*",
        "*/\.conda/*",
        "*/\.venv/*",
        "*/models/*.pth",
        "*/data/*",
        "*/\.pytest_cache/*",
        "*/\.mypy_cache/*",
    ]
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=reload_excludes,
    )
