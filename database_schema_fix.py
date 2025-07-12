import sqlite3
import os
from pathlib import Path

def fix_database_schema():
    """Add missing columns to existing tables"""
    
    # Find the database file
    db_path = Path("data/emergency_response.db")
    if not db_path.exists():
        print("❌ Database file not found. Creating new database...")
        return
    
    print(f"🔧 Fixing database schema: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Add missing columns to crowd_reports table
        missing_columns = [
            ("severity", "INTEGER"),
            ("confidence_score", "REAL"),
            ("ai_analysis", "TEXT"),  # JSON stored as TEXT in SQLite
            ("source", "VARCHAR(50) DEFAULT 'manual'"),
            ("verified", "BOOLEAN DEFAULT 0"),
            ("response_dispatched", "BOOLEAN DEFAULT 0")
        ]
        
        print("📊 Adding missing columns to crowd_reports table...")
        for column_name, column_type in missing_columns:
            try:
                cursor.execute(f"ALTER TABLE crowd_reports ADD COLUMN {column_name} {column_type}")
                print(f"  ✅ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"  ⏭️  Column {column_name} already exists")
                else:
                    print(f"  ❌ Error adding {column_name}: {e}")
        
        # Add missing columns to triage_patients table
        triage_columns = [
            ("priority_score", "INTEGER DEFAULT 5"),
            ("consciousness", "VARCHAR(50) DEFAULT 'unknown'"),
            ("breathing", "VARCHAR(50) DEFAULT 'unknown'")
        ]
        
        print("🏥 Adding missing columns to triage_patients table...")
        for column_name, column_type in triage_columns:
            try:
                cursor.execute(f"ALTER TABLE triage_patients ADD COLUMN {column_name} {column_type}")
                print(f"  ✅ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"  ⏭️  Column {column_name} already exists")
                else:
                    print(f"  ❌ Error adding {column_name}: {e}")
        
        # Create missing tables if they don't exist
        missing_tables = {
            "emergency_reports": """
                CREATE TABLE IF NOT EXISTS emergency_reports (
                    id INTEGER PRIMARY KEY,
                    report_id VARCHAR(50) UNIQUE,
                    type VARCHAR(100) NOT NULL,
                    description TEXT NOT NULL,
                    location VARCHAR(255) NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    priority VARCHAR(20) DEFAULT 'medium',
                    status VARCHAR(20) DEFAULT 'pending',
                    method VARCHAR(20) DEFAULT 'text',
                    reporter VARCHAR(100),
                    evidence_file VARCHAR(255),
                    ai_analysis TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "voice_analyses": """
                CREATE TABLE IF NOT EXISTS voice_analyses (
                    id INTEGER PRIMARY KEY,
                    audio_file_path VARCHAR(255),
                    transcript TEXT,
                    confidence REAL DEFAULT 0.0,
                    urgency_level VARCHAR(20),
                    emergency_type VARCHAR(100),
                    hazards_detected TEXT,
                    emotional_state TEXT,
                    analyst_id VARCHAR(100),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "multimodal_assessments": """
                CREATE TABLE IF NOT EXISTS multimodal_assessments (
                    id INTEGER PRIMARY KEY,
                    assessment_type VARCHAR(50),
                    text_input TEXT,
                    image_path VARCHAR(255),
                    audio_path VARCHAR(255),
                    severity_score REAL DEFAULT 0.0,
                    emergency_type VARCHAR(100),
                    risk_factors TEXT,
                    resource_requirements TEXT,
                    ai_confidence REAL DEFAULT 0.0,
                    analyst_id VARCHAR(100),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        print("📋 Creating missing tables...")
        for table_name, create_sql in missing_tables.items():
            try:
                cursor.execute(create_sql)
                print(f"  ✅ Created/verified table: {table_name}")
            except sqlite3.OperationalError as e:
                print(f"  ❌ Error creating {table_name}: {e}")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("\n✅ Database schema fixed successfully!")
        print("🚀 You can now restart your application: python api.py")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Database Schema Fix Script")
    print("=" * 40)
    
    success = fix_database_schema()
    
    if success:
        print("\n" + "=" * 40)
        print("✅ Schema fix completed!")
        print("Now restart your app: python api.py")
    else:
        print("\n❌ Schema fix failed. You may need to delete the database and start fresh.")