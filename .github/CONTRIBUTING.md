# Contributing to RagaSense

Thank you for your interest in contributing to RagaSense! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)

## Getting Started

### Prerequisites

- Node.js 18 or later
- Python 3.9 or later
- Git
- Lynx Explorer (for frontend testing)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/raga_detector.git
   cd raga_detector
   ```

## Development Setup

### Backend Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database:
   ```bash
   python init_db.py
   ```

4. Run the backend:
   ```bash
   python -m backend.main
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   bun install
   ```

3. Start the development server:
   ```bash
   bun run dev
   ```

4. Test with Lynx Explorer (follow the [frontend README](frontend/README.md) for detailed instructions)

## Code Style

### Python (Backend)

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Keep functions small and focused
- Add docstrings for public functions and classes
- Use meaningful variable and function names

### TypeScript/React (Frontend)

- Use TypeScript for all new code
- Follow ESLint configuration
- Use functional components with hooks
- Keep components small and focused
- Use meaningful variable and function names

### General Guidelines

- Write clear, descriptive commit messages
- Keep changes focused and atomic
- Add comments for complex logic
- Update documentation when changing APIs

## Testing

### Backend Testing

1. Run the test suite:
   ```bash
   python -m pytest tests/
   ```

2. Run the raga detection test:
   ```bash
   python scripts/test_raga_detection.py
   ```

### Frontend Testing

1. Run type checking:
   ```bash
   cd frontend
   bun run type-check
   ```

2. Run linting:
   ```bash
   bun run lint
   ```

3. Test on multiple platforms:
   - Web: `bun run build:web`
   - iOS: `bun run build:ios`
   - Android: `bun run build:android`

## Submitting Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write your code following the style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

```bash
git add .
git commit -m "feat: add new raga detection feature"
```

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Create a Pull Request

- Go to your fork on GitHub
- Click "New Pull Request"
- Select your feature branch
- Fill out the pull request template
- Submit the PR

## Issue Guidelines

### Before Creating an Issue

1. Search existing issues to avoid duplicates
2. Check the documentation for solutions
3. Try to reproduce the issue locally

### Issue Types

- **Bug Report**: Use the bug report template
- **Feature Request**: Use the feature request template
- **Raga Detection Issue**: Use the specialized raga detection template

### Issue Content

- Provide a clear, descriptive title
- Include steps to reproduce (for bugs)
- Add screenshots or logs when relevant
- Specify your environment (OS, browser, etc.)

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] No new warnings are generated
- [ ] Changes are tested on multiple platforms

### PR Content

- Use the pull request template
- Provide a clear description of changes
- Link related issues
- Add screenshots for UI changes
- Include test results

### Review Process

1. Automated checks must pass
2. At least one maintainer must approve
3. All conversations must be resolved
4. Changes may be requested before merging

## Areas for Contribution

### High Priority

- **Raga Detection Accuracy**: Improve ML model performance
- **Audio Processing**: Enhance feature extraction
- **Cross-platform Testing**: Ensure compatibility across devices
- **Performance Optimization**: Improve response times

### Medium Priority

- **UI/UX Improvements**: Enhance user interface
- **Documentation**: Improve guides and tutorials
- **Testing**: Add more comprehensive tests
- **Error Handling**: Better error messages and recovery

### Low Priority

- **Code Refactoring**: Improve code organization
- **Performance Monitoring**: Add metrics and logging
- **Accessibility**: Improve accessibility features
- **Internationalization**: Add multi-language support

## Getting Help

- **Documentation**: Check the [docs](docs/) folder
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Wiki**: Check the project wiki for additional resources

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Contributor hall of fame (if applicable)

Thank you for contributing to RagaSense! Your contributions help make Indian classical music more accessible to everyone.
