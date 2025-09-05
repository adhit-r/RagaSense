#!/usr/bin/env python3
"""
Configuration for RagaSense Backend
"""

import os
from typing import List

class Config:
    """Backend configuration"""
    
    # Convex Configuration
    CONVEX_API_KEY = os.getenv('CONVEX_API_KEY', '')
    CONVEX_URL = os.getenv('CONVEX_URL', 'https://scrupulous-mosquito-279.convex.cloud')
    
    # Backend Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8002))
    DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')
    
    # Authentication
    JWT_SECRET = os.getenv('JWT_SECRET', 'your_jwt_secret_here')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_EXPIRES_IN = os.getenv('JWT_EXPIRES_IN', '24h')
    
    # File Upload
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_AUDIO_TYPES = [
        'audio/mpeg',
        'audio/mp3',
        'audio/wav',
        'audio/flac',
        'audio/ogg'
    ]

# Global config instance
config = Config()
