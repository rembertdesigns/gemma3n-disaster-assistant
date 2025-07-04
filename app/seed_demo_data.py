# Create this file: seed_demo_data.py
# Run this script to populate your database with demo crowd reports

from datetime import datetime, timedelta
import random
from app.database import get_db, engine
from app.models import CrowdReport, TriagePatient, Base

# Sample demo data
demo_reports = [
    {
        "message": "Large tree fallen across Main Street blocking all traffic. Power lines appear damaged. Need immediate cleanup crew.",
        "tone": "Urgent",
        "escalation": "High",
        "user": "Sarah_M",
        "location": "Main St & Oak Ave",
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    {
        "message": "Basement flooding in residential area. Water level rising rapidly. Several families requesting evacuation assistance.",
        "tone": "Frantic", 
        "escalation": "Critical",
        "user": "Emergency_Coord",
        "location": "Riverside District",
        "latitude": 40.7589,
        "longitude": -73.9851
    },
    {
        "message": "Gas smell reported near shopping center. Fire department notified. Area being evacuated as precaution.",
        "tone": "Urgent",
        "escalation": "Critical", 
        "user": "Mike_FD",
        "location": "Central Shopping Plaza",
        "latitude": 40.7282,
        "longitude": -74.0776
    },
    {
        "message": "Traffic light system malfunctioning at major intersection. Causing significant backup and safety concerns.",
        "tone": "Descriptive",
        "escalation": "Moderate",
        "user": "Traffic_Control",
        "location": "5th Ave & Broadway",
        "latitude": 40.7505,
        "longitude": -73.9934
    },
    {
        "message": "Animal shelter reports power outage. Need backup generator for animal life support systems immediately.",
        "tone": "Helpless",
        "escalation": "High",
        "user": "ShelterStaff_Amy",
        "location": "County Animal Shelter",
        "latitude": 40.7361,
        "longitude": -74.0014
    },
    {
        "message": "Update: Road cleared on Main Street. Traffic resuming normal flow. Cleanup crew finished repairs.",
        "tone": "Descriptive",
        "escalation": "Low",
        "user": "Sarah_M",
        "location": "Main St & Oak Ave", 
        "latitude": 40.7128,
        "longitude": -74.0060
    }
]

demo_patients = [
    {
        "name": "John Martinez",
        "age": 34,
        "gender": "male",
        "injury_type": "Laceration",
        "mechanism": "fall",
        "consciousness": "alert",
        "breathing": "normal",
        "heart_rate": 78,
        "bp_systolic": 125,
        "bp_diastolic": 82,
        "respiratory_rate": 16,
        "temperature": 98.6,
        "oxygen_sat": 98,
        "severity": "moderate",
        "triage_color": "yellow",
        "status": "active",
        "notes": "Deep cut on left arm, requires stitches"
    },
    {
        "name": "Maria Rodriguez", 
        "age": 67,
        "gender": "female",
        "injury_type": "Chest Pain",
        "mechanism": "medical",
        "consciousness": "alert",
        "breathing": "labored",
        "heart_rate": 110,
        "bp_systolic": 160,
        "bp_diastolic": 95,
        "respiratory_rate": 22,
        "temperature": 99.1,
        "oxygen_sat": 94,
        "severity": "severe",
        "triage_color": "red",
        "status": "active",
        "notes": "Experiencing acute chest pain, possible cardiac event"
    },
    {
        "name": "David Chen",
        "age": 28,
        "gender": "male", 
        "injury_type": "Sprained Ankle",
        "mechanism": "fall",
        "consciousness": "alert",
        "breathing": "normal",
        "heart_rate": 72,
        "bp_systolic": 118,
        "bp_diastolic": 78,
        "respiratory_rate": 14,
        "temperature": 98.4,
        "oxygen_sat": 99,
        "severity": "mild",
        "triage_color": "green",
        "status": "treated",
        "notes": "Minor ankle sprain, treated and released"
    }
]

def seed_database():
    """Populate database with demo data"""
    print("🌱 Seeding database with demo data...")
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Get database session
    db = next(get_db())
    
    try:
        # Clear existing data (optional - comment out if you want to keep existing data)
        # db.query(CrowdReport).delete()
        # db.query(TriagePatient).delete()
        
        # Add demo crowd reports
        for i, report_data in enumerate(demo_reports):
            # Create timestamp spread over last 2 days
            hours_ago = i * 8 + random.randint(1, 6)
            timestamp = datetime.utcnow() - timedelta(hours=hours_ago)
            
            report = CrowdReport(
                message=report_data["message"],
                tone=report_data["tone"], 
                escalation=report_data["escalation"],
                user=report_data["user"],
                location=report_data["location"],
                timestamp=timestamp.isoformat(),
                latitude=report_data["latitude"],
                longitude=report_data["longitude"]
            )
            db.add(report)
        
        # Add demo patients
        for i, patient_data in enumerate(demo_patients):
            # Create timestamp spread over last day
            hours_ago = i * 4 + random.randint(1, 3)
            created_at = datetime.utcnow() - timedelta(hours=hours_ago)
            
            patient = TriagePatient(
                name=patient_data["name"],
                age=patient_data["age"],
                gender=patient_data["gender"],
                injury_type=patient_data["injury_type"],
                mechanism=patient_data["mechanism"],
                consciousness=patient_data["consciousness"],
                breathing=patient_data["breathing"],
                heart_rate=patient_data["heart_rate"],
                bp_systolic=patient_data["bp_systolic"],
                bp_diastolic=patient_data["bp_diastolic"],
                respiratory_rate=patient_data["respiratory_rate"],
                temperature=patient_data["temperature"],
                oxygen_sat=patient_data["oxygen_sat"],
                severity=patient_data["severity"],
                triage_color=patient_data["triage_color"],
                status=patient_data["status"],
                notes=patient_data["notes"],
                created_at=created_at,
                updated_at=created_at
            )
            db.add(patient)
        
        # Commit all changes
        db.commit()
        
        print("✅ Demo data seeded successfully!")
        print(f"   📋 Added {len(demo_reports)} crowd reports")
        print(f"   🏥 Added {len(demo_patients)} patients")
        print("   🎯 Your dashboards should now show real data!")
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()