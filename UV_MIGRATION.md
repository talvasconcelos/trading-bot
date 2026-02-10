# Migration to uv Package Manager

## Summary
This project has been migrated from pip + requirements.txt to the uv package manager for improved dependency resolution and faster installation times.

## Changes Made
- Created `pyproject.toml` with project metadata and dependencies
- Added dependencies from the original `requirements.txt`
- Created `.python-version` file
- Generated `uv.lock` file for deterministic builds
- Modified `main.py` to have a proper `main()` function for entry points

## Commands Used
- `uv init` - Initialize the project
- `uv sync` - Install dependencies
- `uv run python main.py` - Run the project

## Benefits
- Faster dependency resolution and installation
- Deterministic builds with lock file
- Better virtual environment management
- Modern Python packaging standards compliance