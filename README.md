# Document Classification System

## Overview

The Document Classification System is an advanced tool designed to categorize documents into five security levels: Top Secret, Secret, Confidential, Restricted, and Unclassified. Leveraging state-of-the-art Language Model technology, this system provides instant and accurate classification for various document types.

## Features

- Support for multiple document formats (TXT, PDF, DOCX, DOC)
- Integration with various Language Models (LLMs) including local and cloud-based options
- User-friendly web interface built with Streamlit
- RESTful API for easy integration with other systems
- Batch processing capabilities for directory scanning
- Visualization of classification results
- Export functionality for classification outcomes

## Technology Stack

- Python 3.10+
- FastAPI
- Streamlit
- Plotly
- PyPDF2
- python-docx
- OpenAI API
- Google Generative AI
- Ollama (for local model integration)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/mert-ergun/document-classifier.git
   cd document-classifier
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

   If you don't have Poetry installed, you can install it by following the instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

3. Set up environment variables:
   - Create a `.env` file in the project root and add the following:
     ```
     OPENAI_API_KEY=your_openai_api_key
     GOOGLE_API_KEY=your_google_api_key
     OLLAMA_HOST=http://localhost:11434  # Adjust if your Ollama instance is hosted elsewhere
     ```

## Usage

### Starting the API Server

1. Run the FastAPI server:
   ```
   poetry run uvicorn document_classifier.api:app --host 0.0.0.0 --port 8000
   ```

2. The API will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

### Running the Web Interface

1. Start the Streamlit app:
   ```
   poetry run streamlit run document_classifier/interface.py
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the user interface.

### Using the System

1. Select a classification model from the dropdown menu.
2. Choose between uploading individual files or scanning a directory.
3. For individual files, use the file uploader to select documents.
4. For directory scanning, enter the directory path and click "Scan Directory".
5. View the classification results in the table and pie chart.
6. Download the results as a CSV file if needed.

## API Endpoints

- `POST /classify`: Classify a single document
- `POST /pull_model`: Pull a new model (for local models)
- `GET /tags`: Retrieve available model tags

For detailed API usage, refer to the Swagger documentation at `http://localhost:8000/docs`.


## Docker Support

This project also includes Docker support for easy deployment and consistent environments across different systems.

### Building the Docker Image

To build the Docker image for the Document Classification System, run the following command from the project root:

```bash
docker build -t document-classifier .
```

### Running the Docker Container

Once the image is built, you can run the container using:

```bash
docker run -d -p 8000:8000 -e OPENAI_API_KEY=your_openai_api_key -e GOOGLE_API_KEY=your_google_api_key -e OLLAMA_HOST=http://host.docker.internal:11434 --name doc-classifier document-classifier
```

Replace `your_openai_api_key` and `your_google_api_key` with your actual API keys.

Note: The `OLLAMA_HOST` environment variable is set to `http://host.docker.internal:11434` to allow the container to communicate with Ollama running on the host machine. This works for Docker Desktop on Windows and macOS. For Linux, you might need to use the host's IP address instead.

### Accessing the Application

After running the container, you can access:
- The API at `http://localhost:8000`
- The API documentation at `http://localhost:8000/docs`

### Running the Streamlit Interface with Docker

To run the Streamlit interface, you'll need to create a separate Dockerfile. Create a file named `Dockerfile.streamlit` in the project root with the following content:

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY . /app/

EXPOSE 8501

CMD ["streamlit", "run", "document_classifier/interface.py"]
```

Build the Streamlit Docker image:

```bash
docker build -t document-classifier-streamlit -f Dockerfile.streamlit .
```

Run the Streamlit container:

```bash
docker run -d -p 8501:8501 -e API_URL=http://host.docker.internal:8000 --name doc-classifier-streamlit document-classifier-streamlit
```

You can now access the Streamlit interface at `http://localhost:8501`.

### Docker Compose

For easier management of both services, you can use Docker Compose. Create a `docker-compose.yml` file in the project root:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OLLAMA_HOST=http://host.docker.internal:11434
    volumes:
      - ./docs:/app/docs

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api

networks:
  default:
    name: document-classifier-network
```

To run both services using Docker Compose:

```bash
docker compose up -d
```

This will start both the API and Streamlit services, with the API accessible at `http://localhost:8000` and the Streamlit interface at `http://localhost:8501`.