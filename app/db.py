import sqlite3

DB_PATH = "data/reports.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            location TEXT,
            severity TEXT,
            filename TEXT,
            user TEXT,
            status TEXT,
            image_url TEXT,
            checklist TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_report_metadata(report_data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (id, timestamp, location, severity, filename, user, status, image_url, checklist)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_data["id"],
        report_data["timestamp"],
        report_data["location"],
        report_data["severity"],
        report_data["filename"],
        report_data["user"],
        report_data["status"],
        report_data.get("image_url"),
        ",".join(report_data.get("checklist", []))
    ))
    conn.commit()
    conn.close()