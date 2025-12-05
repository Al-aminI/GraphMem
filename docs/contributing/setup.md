# Development Setup

Set up your development environment for contributing to GraphMem.

## Prerequisites

- Python 3.9+
- Git
- (Optional) Docker for testing with Neo4j/Redis

## Clone the Repository

```bash
git clone https://github.com/Al-aminI/GraphMem.git
cd GraphMem
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

## Install Dependencies

```bash
# Install with all development dependencies
pip install -e ".[dev,all]"
```

## Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=graphmem

# Run specific test file
pytest tests/test_memory.py
```

## Code Formatting

```bash
# Format code
black src/graphmem

# Check formatting
black --check src/graphmem

# Sort imports
isort src/graphmem
```

## Linting

```bash
# Run linter
ruff check src/graphmem

# Fix auto-fixable issues
ruff check --fix src/graphmem
```

## Type Checking

```bash
mypy src/graphmem
```

## Build Documentation

```bash
# Install docs dependencies
pip install -r docs/requirements.txt

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

## Local Services (Docker)

```bash
# Start Neo4j and Redis for testing
docker-compose -f docker-compose.dev.yml up -d

# Stop services
docker-compose -f docker-compose.dev.yml down
```

## Environment Variables

Create a `.env` file for local development:

```bash
# .env
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j://localhost:7687
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379
```

