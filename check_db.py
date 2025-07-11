import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print database configuration
print("Database Configuration:")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")
print(f"DB_USER: {os.getenv('DB_USER')}")
print(f"DB_PASSWORD: {'*' * len(os.getenv('DB_PASSWORD', ''))}" if os.getenv('DB_PASSWORD') else "DB_PASSWORD: Not set")
print(f"DB_HOST: {os.getenv('DB_HOST')}")
print(f"DB_PORT: {os.getenv('DB_PORT')}")
print(f"DB_NAME: {os.getenv('DB_NAME')}")

# Try to connect to the database
if os.getenv('DATABASE_URL'):
    try:
        import psycopg2
        from urllib.parse import urlparse
        
        # Parse the database URL
        db_url = os.getenv('DATABASE_URL')
        print(f"\nAttempting to connect to: {db_url}")
        
        # Connect to the database
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Get database version
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"\nConnected to PostgreSQL {db_version[0]}")
        
        # List all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        print("\nTables in the database:")
        for table in tables:
            print(f"- {table[0]}")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"\nError connecting to the database: {e}")
        print("Make sure the database is running and the connection details are correct.")
