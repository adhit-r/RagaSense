#!/usr/bin/env python
"""
Management script for the Raga Detector application.

This script provides command-line utilities for managing the application,
including running the development server, initializing the database, and
running database migrations.
"""
import os
import sys
import click
from flask_migrate import Migrate

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend import create_app, db
from backend.models.raga import Raga

# Create the Flask application
app = create_app()
migrate = Migrate(app, db)

def make_shell_context():
    """
    Add objects to the shell context for easier interactive development.
    These objects will be available in the Flask shell.
    """
    return {
        'app': app,
        'db': db,
        'Raga': Raga,
    }

# Register the shell context processor
app.shell_context_processor(make_shell_context)

@app.cli.command("init-db")
@click.option('--drop', is_flag=True, 
              help='Drop existing database tables before creating new ones.')
def init_db(drop):
    """Initialize the database."""
    if drop:
        click.confirm('This will delete all data in the database. Are you sure?', 
                     abort=True)
        db.drop_all()
        click.echo('Dropped all tables.')
    
    db.create_all()
    click.echo('Initialized the database.')

@app.cli.command("import-ragas")
@click.argument('file_path', required=False, type=click.Path(exists=True))
def import_ragas(file_path):
    """Import ragas from a JSON file."""
    from backend.scripts.import_ragas import import_ragas as _import_ragas
    
    if file_path:
        result = _import_ragas(file_path)
    else:
        # Use the default sample data
        default_path = os.path.join('backend', 'data', 'ragas_sample.json')
        if not os.path.exists(default_path):
            click.echo(f"Error: Default raga file not found at {default_path}")
            sys.exit(1)
        result = _import_ragas(default_path)
    
    if not result:
        sys.exit(1)

if __name__ == '__main__':
    app.run()
