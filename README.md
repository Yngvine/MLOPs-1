# MLOps-Lab1

## Overview

This project is the first lab in a series of MLOps assignments focused on building a Continuous Integration (CI) pipeline using GitHub Actions. The ultimate goal (Lab 3) is to create a tool for image classification using deep learning methods. This first lab establishes the foundational structure and basic logic for the image classification pipeline.

## Project Structure

```
MLOps-Lab1/
├── api/
│   ├── __init__.py
│   └── fastapi_main.py       # FastAPI implementation
├── cli/
│   ├── __init__.py
│   └── cli.py                # Command Line Interface
├── mylib/
│   ├── __init__.py
│   ├── calculator.py
│   └── logic.py              # Core image processing logic
├── templates/
│   └── home.html             # Web interface templates
├── tests/
│   ├── __init__.py
│   ├── test_cli.py           # CLI tests
│   └── test_fastapi_main.py  # API tests
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── uv.txt
```

## Objectives

### Core Functionality

1. **Image Classification Module** (`mylib/logic.py`)
   - Implement a prediction method that randomly selects a class from a predefined set of class names
   - This serves as a placeholder for future deep learning implementation

2. **Image Preprocessing** (`mylib/logic.py`)
   - Implement an image resizing method to standardize image dimensions
   - Additional preprocessing methods can be added as needed

3. **Command Line Interface** (`cli/cli.py`)
   - Provide CLI access to the core functionalities
   - Enable user interaction with image processing and classification features

4. **REST API** (`api/fastapi_main.py`)
   - Implement a FastAPI-based web service
   - Expose the core functionalities through HTTP endpoints

### Testing

All components must include comprehensive tests:
- `tests/test_cli.py` - CLI functionality tests
- `tests/test_fastapi_main.py` - API endpoint tests
- Additional test files as needed for core logic

## Getting Started

### Prerequisites

- Python 3.x
- pip or uv package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MLOps-Lab1.git
cd MLOps-Lab1

# Install dependencies
make install
```

### Usage

```bash
# The api can be deployed via
uv run python -m api.fastapi_main
```

## Continuous Integration

This project uses GitHub Actions for CI/CD pipeline automation, ensuring code quality and automated testing on each commit.

## Future Development

- **Lab 2**: Enhancement of preprocessing and feature extraction
- **Lab 3**: Integration of deep learning models for actual image classification

## License

See LICENSE file for details.

## Contributing

This is an academic project. Contributions follow the course guidelines.
