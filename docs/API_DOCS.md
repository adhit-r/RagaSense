# Raga Detection API Documentation

## Base URL
```
http://localhost:5000/api/v1
```

## Authentication
All endpoints are currently public and don't require authentication.

## Rate Limiting
- 100 requests per minute per IP address
- 1000 requests per hour per IP address

## Error Responses
All error responses follow this format:
```json
{
  "success": false,
  "error": "Error type",
  "message": "Human-readable error message"
}
```

## Endpoints

### Get All Ragas
```
GET /ragas
```

**Query Parameters:**
- `page` - Page number (default: 1)
- `per_page` - Items per page (default: 20, max: 100)
- `search` - Search query to filter ragas by name or description

**Example Request:**
```
GET /api/v1/ragas?page=1&per_page=10&search=yaman
```

**Example Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Yaman",
      "alternate_names": ["Iman", "Kalyani"],
      "arohana": ["S", "R", "G", "M", "P", "D", "N", "S'"],
      "avarohana": ["S'", "N", "D", "P", "M", "G", "R", "S"],
      "time": "Evening",
      "mood": "Devotional, Peaceful, Romantic",
      "description": "Yaman is a popular raga in the Hindustani classical tradition..."
    }
  ],
  "pagination": {
    "total": 100,
    "pages": 10,
    "current_page": 1,
    "per_page": 10,
    "has_next": true,
    "has_prev": false
  }
}
```

### Get a Single Raga
```
GET /ragas/:id
```

**URL Parameters:**
- `id` - The ID of the raga to retrieve

**Example Request:**
```
GET /api/v1/ragas/1
```

**Example Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "Yaman",
    "alternate_names": ["Iman", "Kalyani"],
    "arohana": ["S", "R", "G", "M", "P", "D", "N", "S'"],
    "avarohana": ["S'", "N", "D", "P", "M", "G", "R", "S"],
    "time": "Evening",
    "mood": "Devotional, Peaceful, Romantic",
    "description": "Yaman is a popular raga in the Hindustani classical tradition...",
    "characteristic_phrases": ["N R G M D N S'", "S' N D P M G R S"],
    "similar_ragas": [
      {"id": 2, "name": "Bhoopali", "similarity": 0.85},
      {"id": 3, "name": "Deshkar", "similarity": 0.78}
    ]
  }
}
```

### Search Ragas
```
GET /ragas/search
```

**Query Parameters:**
- `q` - Search query string (required)
- `limit` - Maximum number of results (default: 10, max: 50)

**Example Request:**
```
GET /api/v1/ragas/search?q=yaman&limit=5
```

**Example Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Yaman",
      "alternate_names": ["Iman", "Kalyani"],
      "description": "Yaman is a popular raga in the Hindustani classical tradition..."
    }
  ],
  "count": 1
}
```

### Analyze Audio
```
POST /analyze
Content-Type: multipart/form-data
```

**Form Data:**
- `file` - Audio file to analyze (required)

**Supported Audio Formats:**
- WAV, MP3, OGG, FLAC, M4A, AAC

**Example Request:**
```
POST /api/v1/analyze
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="sample.mp3"
Content-Type: audio/mpeg

[binary data]
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

**Example Response (200 OK):**
```json
{
  "success": true,
  "analysis": {
    "detected_raga": {
      "id": 1,
      "name": "Yaman",
      "confidence": 0.92,
      "alternate_names": ["Iman", "Kalyani"],
      "arohana": ["S", "R", "G", "M", "P", "D", "N", "S'"],
      "avarohana": ["S'", "N", "D", "P", "M", "G", "R", "S"],
      "time": "Evening",
      "mood": "Devotional, Peaceful, Romantic"
    },
    "features": {
      "pitch_contour": [
        {"time": 0.0, "pitch": 220.0},
        {"time": 0.5, "pitch": 246.94}
      ],
      "tempo": 72.5,
      "key": "C",
      "scale": "Major"
    },
    "similar_ragas": [
      {"id": 2, "name": "Bhoopali", "similarity": 0.85},
      {"id": 3, "name": "Deshkar", "similarity": 0.78}
    ]
  }
}
```

## Error Responses

### 400 Bad Request
```json
{
  "success": false,
  "error": "Invalid file type",
  "message": "Allowed file types are: wav, mp3, ogg, flac, m4a, aac"
}
```

### 404 Not Found
```json
{
  "success": false,
  "error": "Not Found",
  "message": "The requested raga was not found"
}
```

### 500 Internal Server Error
```json
{
  "success": false,
  "error": "Internal Server Error",
  "message": "An unexpected error occurred"
}
```
