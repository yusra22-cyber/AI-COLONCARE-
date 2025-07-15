import sqlite3

def create_db():
    # Connect to SQLite database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create the users table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()

def add_user(first_name, last_name, email, password):
    # Connect to SQLite database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Insert new user into users table
    c.execute('''
        INSERT INTO users (first_name, last_name, email, password)
        VALUES (?, ?, ?, ?)
    ''', (first_name, last_name, email, password))

    # Commit changes and close connection
    conn.commit()
    conn.close()

def check_user(email, password):
    # Connect to SQLite database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Check if a user exists with the given email and password
    c.execute('''
        SELECT * FROM users WHERE email = ? AND password = ?
    ''', (email, password))

    user = c.fetchone()

    # Close connection
    conn.close()

    return user
