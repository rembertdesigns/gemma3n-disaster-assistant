import sqlite3
from datetime import datetime

DB_PATH = "data/reports.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# =========================
# 🏗️ Schema Initialization
# =========================

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Primary reports table
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
            checklist TEXT,
            latitude REAL,
            longitude REAL
        )
    """)

    # Crowd reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_reports (
            id TEXT PRIMARY KEY,
            message TEXT,
            location TEXT,
            user TEXT,
            timestamp TEXT,
            sentiment TEXT,
            tone TEXT,
            escalation TEXT
        )
    """)

    # 🆕 Triage patients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS triage_patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            injury_type TEXT,
            severity TEXT,
            vitals TEXT,
            notes TEXT,
            triage_color TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()

# =========================
# 🔁 Migrations
# =========================

def run_migrations():
    """Safely add missing columns to existing tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN latitude REAL;")
        cursor.execute("ALTER TABLE reports ADD COLUMN longitude REAL;")
        conn.commit()
        print("✅ Added latitude and longitude columns to reports table.")
    except Exception as e:
        print(f"⚠️ Skipping migration (maybe already applied): {e}")
    conn.close()

# =========================
# 📝 Report Metadata
# =========================

def save_report_metadata(report_data: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (
            id, timestamp, location, severity, filename,
            user, status, image_url, checklist, latitude, longitude
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_data["id"],
        report_data["timestamp"],
        report_data["location"],
        report_data["severity"],
        report_data["filename"],
        report_data["user"],
        report_data["status"],
        report_data.get("image_url"),
        ",".join(report_data.get("checklist", [])),
        report_data.get("latitude"),
        report_data.get("longitude")
    ))
    conn.commit()
    conn.close()

def get_all_reports(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports ORDER BY timestamp DESC")
    return cursor.fetchall()

def get_report_by_id(conn, report_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
    return cursor.fetchone()

def get_dashboard_stats(conn):
    cursor = conn.cursor()
    stats = {}

    cursor.execute("SELECT COUNT(*) FROM reports")
    stats["total_reports"] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(CAST(severity AS REAL)) FROM reports")
    stats["avg_severity"] = round(cursor.fetchone()[0] or 0, 2)

    cursor.execute("SELECT user, COUNT(*) FROM reports GROUP BY user")
    stats["reports_per_user"] = cursor.fetchall()

    cursor.execute("SELECT status, COUNT(*) FROM reports GROUP BY status")
    stats["status_counts"] = cursor.fetchall()

    return stats

# =========================
# 📣 Crowd Reports
# =========================

def save_crowd_report(report: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO crowd_reports (id, message, location, user, timestamp, sentiment, tone, escalation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report["id"],
        report["message"],
        report["location"],
        report["user"],
        report["timestamp"],
        report["sentiment"],
        report["tone"],
        report["escalation"]
    ))
    conn.commit()
    conn.close()

def get_crowd_reports(filters: dict = None):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM crowd_reports"
    clauses = []
    params = []

    if filters:
        if "tone" in filters:
            clauses.append("tone = ?")
            params.append(filters["tone"])
        if "escalation" in filters:
            clauses.append("escalation = ?")
            params.append(filters["escalation"])

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY timestamp DESC"

    cursor.execute(query, tuple(params))
    results = cursor.fetchall()
    conn.close()
    return results

def get_crowd_report_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    stats = {}

    cursor.execute("SELECT COUNT(*) FROM crowd_reports")
    stats["total_crowd_reports"] = cursor.fetchone()[0]

    cursor.execute("SELECT tone, COUNT(*) FROM crowd_reports GROUP BY tone")
    stats["tone_distribution"] = cursor.fetchall()

    cursor.execute("SELECT escalation, COUNT(*) FROM crowd_reports GROUP BY escalation")
    stats["escalation_distribution"] = cursor.fetchall()

    return stats

# =========================
# 🏥 Triage Patients
# =========================

def save_triage_patient(data: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO triage_patients (
            id, name, age, injury_type, severity,
            vitals, notes, triage_color, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["id"],
        data["name"],
        data["age"],
        data["injury_type"],
        data["severity"],
        data["vitals"],
        data["notes"],
        data["triage_color"],
        data["timestamp"]
    ))
    conn.commit()
    conn.close()

def get_all_triage_patients():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM triage_patients ORDER BY timestamp DESC")
    results = cursor.fetchall()
    conn.close()
    return results

def get_triage_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    stats = {}

    cursor.execute("SELECT COUNT(*) FROM triage_patients")
    stats["total_patients"] = cursor.fetchone()[0]

    cursor.execute("SELECT triage_color, COUNT(*) FROM triage_patients GROUP BY triage_color")
    stats["color_distribution"] = cursor.fetchall()

    cursor.execute("SELECT severity, COUNT(*) FROM triage_patients GROUP BY severity")
    stats["severity_distribution"] = cursor.fetchall()

    return stats