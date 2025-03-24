# Contributing to Conditional Diffusion Classifier

Thank you for your interest in contributing to this project! This document provides guidelines and steps for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/ahmed-3m/conditional_diffusion.git
cd conditional_diffusion
```

3. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

4. Install development dependencies:
```bash
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and return values
- Add docstrings to all functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable names

## Testing

Before submitting a pull request:
1. Run the test suite:
```bash
python -m pytest tests/
```

2. Ensure all tests pass
3. Add new tests for any new functionality

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the requirements.txt if you add new dependencies
3. Create a Pull Request with a clear description of your changes
4. Link any related issues
5. Wait for review and address any feedback

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Project Structure

When adding new features:
- Put new model architectures in `utils/model.py`
- Add dataset handling code to `utils/datasets.py`
- Keep the main training logic in `main.py`
- Add new utilities to the `utils/` directory

## Questions?

Feel free to open an issue for any questions or concerns about contributing. 