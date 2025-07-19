# Dolphin FastAPI Service

A FastAPI service for document processing with sliding window approach, providing OpenAI API compatible endpoints for PDF analysis with semantic overlap detection.

## Features

- 🔄 **Sliding Window Processing**: Process PDFs with overlapping windows (1-2, 2-3, 3-4, etc.)
- 🧠 **Semantic Analysis**: Intelligent overlap detection and content merging using sentence transformers
- 🔗 **OpenAI API Compatible**: `/v1/chat/completions` endpoint compatible with OpenAI API format
- ⚡ **Async Processing**: Background job processing for large documents
- 🎯 **Cross-Page Intelligence**: Identifies and merges content spanning multiple pages
- 📊 **Multiple Output Formats**: JSON, Markdown, HTML with comprehensive metadata
- 🚀 **Enterprise Ready**: Scalable architecture with monitoring and caching

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Dolphin model files (see main README)

### Installation

1. **Install dependencies**:
   ```bash
   cd scanner
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp config/env.example .env
   # Edit .env file with your settings
   ```

3. **Check installation**:
   ```bash
   python start_service.py --check-only
   ```

4. **Start the service**:
   ```bash
   # Development mode
   python start_service.py --reload

   # Production mode
   python start_service.py --workers 4
   ```

The service will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## API Endpoints

### OpenAI Compatible Chat Completions

**POST** `/v1/chat/completions`

Process documents with prompts using OpenAI API format:

```python
import requests

# For document with file upload
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "dolphin-pdf-processor",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this document and extract key information"},
                {"type": "document", "document": file_content_bytes}
            ]
        }
    ],
    "window_size": 2,
    "overlap": 1,
    "semantic_threshold": 0.8
})

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Direct PDF Processing

**POST** `/process-pdf`

Upload and process PDF files directly:

```python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process-pdf",
        files={"file": f},
        data={
            "prompt": "Extract and analyze the content of this document",
            "window_size": 2,
            "overlap": 1,
            "output_format": "json"
        }
    )

job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{job_id}")
print(status.json())

# Get results when completed
results = requests.get(f"http://localhost:8000/results/{job_id}")
print(results.json())
```

### Health Check

**GET** `/health`

Check service health and model status:

```bash
curl http://localhost:8000/health
```

## Configuration

### Environment Variables

Key configuration options (see `config/env.example` for full list):

```bash
# Model paths
DOLPHIN_MODEL_CONFIG_PATH=../config/Dolphin.yaml
DOLPHIN_MODEL_CHECKPOINT_PATH=../checkpoints

# Processing settings
DOLPHIN_DEFAULT_WINDOW_SIZE=2
DOLPHIN_DEFAULT_OVERLAP=1
DOLPHIN_DEFAULT_SEMANTIC_THRESHOLD=0.8

# Performance
DOLPHIN_ENABLE_GPU=true
DOLPHIN_MAX_BATCH_SIZE=4
DOLPHIN_ENABLE_CACHING=true

# File limits
DOLPHIN_MAX_FILE_SIZE=52428800  # 50MB
```

### Processing Parameters

- **window_size**: Number of pages per sliding window (default: 2)
- **overlap**: Pages to overlap between windows (default: 1)  
- **semantic_threshold**: Similarity threshold for overlap detection (0.0-1.0)
- **output_format**: Response format (json, markdown, html)

## Advanced Usage

### Batch Processing

Process multiple documents in parallel:

```python
documents = [
    {"filename": "doc1.pdf", "content": doc1_bytes},
    {"filename": "doc2.pdf", "content": doc2_bytes}
]

response = requests.post("http://localhost:8000/batch-process", json={
    "documents": documents,
    "prompt": "Analyze these documents",
    "merge_results": True
})
```

### Page Range Processing

Process specific page ranges:

```python
response = requests.post("http://localhost:8000/process-pages", json={
    "start_page": 5,
    "end_page": 10,
    "window_size": 2,
    "prompt": "Extract content from these pages"
})
```

### Custom Semantic Analysis

Fine-tune semantic analysis parameters:

```python
response = requests.post("http://localhost:8000/process-pdf", json={
    "semantic_threshold": 0.85,  # Higher threshold for stricter matching
    "merge_tables": True,        # Merge cross-page tables
    "preserve_formulas": True    # Handle formula continuity
})
```

## Response Format

### Processing Result Structure

```json
{
  "document_id": "uuid",
  "total_pages": 10,
  "processing_time": 25.4,
  "window_count": 9,
  "overlap_detections": 8,
  "merge_operations": 3,
  "merged_content": {
    "text_content": "Complete merged document text...",
    "structured_data": {
      "total_elements": 45,
      "merged_elements": 3,
      "discrete_paragraphs": 42
    },
    "tables": [...],
    "formulas": [...],
    "images": [...]
  },
  "semantic_relationships": [...],
  "cross_page_elements": [...],
  "confidence_scores": {
    "processing_success_rate": 0.95,
    "semantic_analysis_confidence": 0.87,
    "overall_confidence": 0.91
  }
}
```

### Semantic Relationships

Detected relationships between content across pages:

```json
{
  "source_window": 1,
  "target_window": 2,
  "similarity_score": 0.92,
  "relation_type": "exact",
  "source_element_id": "elem_123",
  "target_element_id": "elem_456"
}
```

### Cross-Page Elements

Elements spanning multiple pages:

```json
{
  "element_id": "merged_1_2_789",
  "element_type": "table",
  "start_page": 1,
  "end_page": 2,
  "merged_content": "Complete table content...",
  "confidence_score": 0.94
}
```

## Docker Deployment

### Build Image

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_service.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Run Container

```bash
docker build -t dolphin-api .
docker run -p 8000:8000 -v /path/to/models:/app/models dolphin-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  dolphin-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DOLPHIN_ENABLE_GPU=false
      - DOLPHIN_REDIS_HOST=redis
    volumes:
      - ./models:/app/models
    depends_on:
      - redis

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## Performance Optimization

### GPU Acceleration

Ensure CUDA is available for optimal performance:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Batch Size Tuning

Adjust batch size based on GPU memory:

```bash
# For 8GB GPU
DOLPHIN_MAX_BATCH_SIZE=4

# For 16GB GPU  
DOLPHIN_MAX_BATCH_SIZE=8

# For 24GB+ GPU
DOLPHIN_MAX_BATCH_SIZE=16
```

### Caching

Enable Redis caching for better performance:

```bash
DOLPHIN_ENABLE_CACHING=true
DOLPHIN_REDIS_HOST=localhost
DOLPHIN_CACHE_TTL=3600
```

## Monitoring

### Health Metrics

Monitor service health:

```bash
curl http://localhost:8000/health?include_model_status=true&include_gpu_status=true
```

### Processing Statistics

Get detailed processing statistics:

```python
response = requests.get("http://localhost:8000/metrics")
metrics = response.json()

print(f"Total requests: {metrics['requests_total']}")
print(f"Average processing time: {metrics['average_processing_time']:.2f}s")
print(f"GPU utilization: {metrics['gpu_utilization']:.1f}%")
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model files
   python start_service.py --check-only
   
   # Verify paths in .env file
   DOLPHIN_MODEL_CONFIG_PATH=../config/Dolphin.yaml
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   DOLPHIN_MAX_BATCH_SIZE=2
   
   # Or disable GPU
   DOLPHIN_ENABLE_GPU=false
   ```

3. **File Upload Limits**
   ```bash
   # Increase file size limit
   DOLPHIN_MAX_FILE_SIZE=104857600  # 100MB
   ```

### Debug Mode

Enable debug logging:

```bash
python start_service.py --log-level DEBUG --reload
```

### Performance Profiling

Profile processing performance:

```python
import requests
import time

start_time = time.time()
response = requests.post("http://localhost:8000/process-pdf", ...)
end_time = time.time()

print(f"Total time: {end_time - start_time:.2f}s")
print(f"Service processing time: {response.json()['processing_time']:.2f}s")
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Code Quality

```bash
# Format code
black scanner/

# Type checking
mypy scanner/

# Linting
flake8 scanner/
```

## API Reference

For complete API documentation, visit the interactive docs when the service is running:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 