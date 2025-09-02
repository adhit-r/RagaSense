#!/usr/bin/env python3
"""
Convex Configuration for RagaSense Backend
Uses existing production Convex functions
"""

import os
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import uuid

class ConvexClient:
    """Convex client using existing production functions"""
    
    def __init__(self):
        self.base_url = "https://scrupulous-mosquito-279.convex.cloud"
        self.api_key = os.getenv('CONVEX_API_KEY')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to Convex"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            elif method == 'PUT':
                response = requests.put(url, headers=self.headers, json=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Convex API error: {e}")
            return {"error": str(e)}
    
    def upload_file(self, file_data: bytes, filename: str, content_type: str, user_id: str = "default_user") -> Dict:
        """Upload file using existing createFile function"""
        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Create file metadata
            file_metadata = {
                "fileId": file_id,
                "filename": filename,
                "contentType": content_type,
                "size": len(file_data),
                "uploadedAt": datetime.utcnow().isoformat(),
                "hash": hashlib.md5(file_data).hexdigest(),
                "userId": user_id,
                "type": "audio",
                "status": "uploaded"
            }
            
            # Use existing createFile function
            result = self._make_request(
                "/api/actions/files/createFile",
                method="POST",
                data=file_metadata
            )
            
            if "error" not in result:
                result["fileId"] = file_id
                result["filename"] = filename
                result["contentType"] = content_type
                result["size"] = len(file_data)
                result["uploadedAt"] = file_metadata["uploadedAt"]
                result["hash"] = file_metadata["hash"]
            
            return result
            
        except Exception as e:
            print(f"❌ File upload error: {e}")
            return {"error": str(e)}
    
    def classify_audio(self, file_id: str, audio_data: bytes, user_id: str = "default_user") -> Dict:
        """Use existing classifyAudio function"""
        try:
            # Prepare audio data for classification
            classification_data = {
                "fileId": file_id,
                "audioData": audio_data.hex(),  # Convert to hex for JSON
                "userId": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            result = self._make_request(
                "/api/actions/ragaDetection/classifyAudio",
                method="POST",
                data=classification_data
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Audio classification error: {e}")
            return {"error": str(e)}
    
    def get_user_files(self, user_id: str) -> List[Dict]:
        """Get files using existing getUserFiles function"""
        try:
            result = self._make_request(
                f"/api/actions/files/getUserFiles?userId={user_id}",
                method="GET"
            )
            
            return result.get("files", [])
            
        except Exception as e:
            print(f"❌ Get user files error: {e}")
            return []
    
    def get_classification_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get classification history using existing function"""
        try:
            result = self._make_request(
                f"/api/actions/classification/getClassificationHistory?userId={user_id}&limit={limit}",
                method="GET"
            )
            
            return result.get("classifications", [])
            
        except Exception as e:
            print(f"❌ Get classification history error: {e}")
            return []
    
    def search_ragas(self, search_term: str, tradition: Optional[str] = None) -> List[Dict]:
        """Search ragas using existing searchRagas function"""
        try:
            params = {"searchTerm": search_term}
            if tradition:
                params["tradition"] = tradition
            
            result = self._make_request(
                "/api/actions/ragas/searchRagas",
                method="POST",
                data=params
            )
            
            return result.get("ragas", [])
            
        except Exception as e:
            print(f"❌ Raga search error: {e}")
            return []
    
    def get_all_ragas(self, tradition: Optional[str] = None) -> List[Dict]:
        """Get all ragas using existing function"""
        try:
            if tradition:
                result = self._make_request(
                    f"/api/actions/ragas/getRagasByTradition?tradition={tradition}",
                    method="GET"
                )
            else:
                result = self._make_request(
                    "/api/actions/ragas/getAllRagas",
                    method="GET"
                )
            
            return result.get("ragas", [])
            
        except Exception as e:
            print(f"❌ Get ragas error: {e}")
            return []
    
    def get_file_by_id(self, file_id: str) -> Dict:
        """Get file details using existing getFileById function"""
        try:
            result = self._make_request(
                f"/api/actions/files/getFileById?fileId={file_id}",
                method="GET"
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Get file error: {e}")
            return {"error": str(e)}
    
    def delete_file(self, file_id: str, user_id: str) -> Dict:
        """Delete file using existing deleteFile function"""
        try:
            result = self._make_request(
                "/api/actions/files/deleteFile",
                method="POST",
                data={
                    "fileId": file_id,
                    "userId": user_id
                }
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Delete file error: {e}")
            return {"error": str(e)}

# Initialize Convex client
convex_client = ConvexClient()
