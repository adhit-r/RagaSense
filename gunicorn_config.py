"""Gunicorn configuration for the Raga Detection System."""
import multiprocessing
import os

# Server socket
bind = os.getenv('GUNICORN_BIND', '127.0.0.1:8000')
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'gevent'
worker_connections = 1000
timeout = 300
keepalive = 2

# Process naming
proc_name = 'raga_detector'

# Logging
accesslog = 'logs/gunicorn-access.log'
errorlog = 'logs/gunicorn-error.log'
loglevel = 'info'

# SSL (uncomment if not using reverse proxy)
# keyfile = 'certs/privkey.pem'
# certfile = 'certs/fullchain.pem'

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Development
reload = os.getenv('FLASK_ENV') == 'development'
reload_engine = 'auto'

# Process management
preload_app = True
daemon = False

def on_starting(server):
    """Log when the server starts."""
    server.log.info("Starting Raga Detection System")

def on_reload(server):
    """Log when the server reloads."""
    server.log.info("Reloading Raga Detection System")

def post_fork(server, worker):
    """Reset process title after fork."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def worker_abort(worker):
    """Log when a worker is aborted."""
    worker.log.info("Worker aborted")
