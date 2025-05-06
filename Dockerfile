# Use slim Python 3.9 as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependency specs
COPY pyproject.toml poetry.lock ./

# Install Poetry, disable venv creation, install only prod deps
RUN pip install poetry \
 && poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --without dev --no-root

# Copy your application code
COPY . .

# Make sure your orchestration scripts are executable
RUN chmod +x scripts/master.py \
 && chmod +x scripts/verify_setup.sh \
 && chmod +x scripts/rollback_models.sh \
 && chmod +x scripts/update_symbols.py

# Default command: run your master orchestrator
CMD ["poetry", "run", "python", "scripts/master.py"]