#!/usr/bin/env python3
"""
Simple HTTP server for RagaSense website
Run with: python server.py
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8081

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Change to website directory
    website_dir = Path(__file__).parent
    os.chdir(website_dir)
    
    # Create server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"RagaSense Website Server")
        print(f"Server running at http://localhost:{PORT}")
        print(f"Serving files from: {website_dir}")
        print(f"Terminal version: http://localhost:{PORT}/index-terminal.html")
        print(f"Press Ctrl+C to stop the server")
        
        # Open browser
        webbrowser.open(f'http://localhost:{PORT}')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    main()
