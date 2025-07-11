#!/usr/bin/env python3
# quick_migration.py - Run this to update your database

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def run_migration():
    """Quick migration to create missing tables"""
    
    print("🔧 Quick Database Migration")
    print("=" * 40)
    
    try:
        # Import after adding to path
        from app.database import engine
        from app.models import Base
        
        print("✅ Successfully imported database modules")
        
        # Create all tables
        print("📊 Creating/updating database tables...")
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database tables created successfully!")
        
        # Test database connection
        from app.database import get_db
        from app.models import User, CrowdReport, TriagePatient
        
        db = next(get_db())
        
        # Test basic queries
        user_count = db.query(User).count()
        report_count = db.query(CrowdReport).count()
        patient_count = db.query(TriagePatient).count()
        
        print(f"📋 Database Status:")
        print(f"   Users: {user_count}")
        print(f"   Reports: {report_count}")
        print(f"   Patients: {patient_count}")
        
        # Create default admin user if no users exist
        if user_count == 0:
            print("👤 Creating default admin user...")
            
            import hashlib
            hashed_password = hashlib.sha256("admin".encode()).hexdigest()
            
            admin_user = User(
                username="admin",
                email="admin@emergency.local",
                hashed_password=hashed_password,
                role="admin",
                is_active=True
            )
            
            db.add(admin_user)
            db.commit()
            
            print("✅ Default admin user created:")
            print("   Username: admin")
            print("   Password: admin")
        
        db.close()
        
        print("\n" + "=" * 40)
        print("🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("1. Run: python api.py")
        print("2. Visit: http://localhost:8000")
        print("3. Login with admin/admin")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        print("\nThis might be normal if some tables already exist.")
        print("Try running your application anyway: python api.py")
        return False

if __name__ == "__main__":
    run_migration()
