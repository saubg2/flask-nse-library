import sqlite3
import os

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'flask_app', 'nselib_data.db')

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

conn = get_db_connection()
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables and their fields in the database:")
if tables:
    for table in tables:
        table_name = table['name']
        print(f"\nTable: {table_name}")
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        if columns:
            for col in columns:
                print(f"  - {col['name']} ({col['type']})")
        else:
            print("  (No columns found)")
else:
    print("No tables found in the database.")

conn.close()