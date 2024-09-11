FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* /app/

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy project
COPY . /app/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "document_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]