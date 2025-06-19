# Qwen 2.5-VL API Documentation

## REST API Endpoints

### Chat Completions

**POST** `/v1/chat/completions`

Create a chat completion with vision capabilities.

#### Request Body

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."}
                }
            ]
        }
    ],
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
}
```

#### Response

```json
{
    "response": "I can see a beautiful sunset over the ocean...",
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    },
    "model": "qwen-2.5-vl",
    "created": 1640995200
}
```

### Image Analysis

**POST** `/v1/analyze/image`

Analyze an uploaded image file.

#### Request

- **Content-Type**: `multipart/form-data`
- **file**: Image file
- **prompt**: Analysis prompt (optional)

#### Response

```json
{
    "analysis": "This image shows..."
}
```

### Health Check

**GET** `/health`

Check service health and model status.

#### Response

```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_info": {
        "model_type": "qwen2_5_vl",
        "total_parameters": 3000000000,
        "device": "cuda",
        "memory_usage": "8.2GB"
    }
}
```

### Load Model

**POST** `/load_model`

Load a specific model variant.

#### Request Body

```json
{
    "variant": "qwen-2.5-vl-7b"
}
```

#### Response

```json
{
    "status": "success",
    "model": "qwen-2.5-vl-7b"
}
```

## Authentication

All endpoints require an API key when authentication is enabled:

```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limiting

- 60 requests per minute per API key
- 1000 requests per hour per API key

## Error Responses

```json
{
    "detail": "Error description",
    "status_code": 400
}
```
