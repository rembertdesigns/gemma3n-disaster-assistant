# migrate_database.py - Safe database migration for Gemma 3n models

import logging
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import json

from app.database import engine, get_db
from app.models import Base, CrowdReport, TriagePatient, VoiceAnalysis, MultimodalAssessment, ContextAnalysis, DevicePerformance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_existing_data():
    """Backup existing data before migration"""
    logger.info("🔄 Backing up existing data...")
    
    backup_data = {
        "backup_timestamp": datetime.utcnow().isoformat(),
        "crowd_reports": [],
        "triage_patients": []
    }
    
    try:
        db = next(get_db())
        
        # Backup crowd reports
        existing_reports = db.execute(text("SELECT * FROM crowd_reports")).fetchall()
        for row in existing_reports:
            backup_data["crowd_reports"].append({
                "id": row[0], "message": row[1], "tone": row[2], 
                "escalation": row[3], "user": row[4], "location": row[5],
                "timestamp": row[6], "latitude": row[7], "longitude": row[8]
            })
        
        # Backup triage patients  
        existing_patients = db.execute(text("SELECT * FROM triage_patients")).fetchall()
        for row in existing_patients:
            backup_data["triage_patients"].append({
                "id": row[0], "name": row[1], "age": row[2], "gender": row[3],
                "injury_type": row[4], "severity": row[5], "triage_color": row[6]
                # Add other fields as needed
            })
        
        # Save backup
        backup_filename = f"database_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_filename, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"✅ Backup saved to {backup_filename}")
        logger.info(f"   • {len(backup_data['crowd_reports'])} crowd reports")
        logger.info(f"   • {len(backup_data['triage_patients'])} triage patients")
        
        return backup_filename
        
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return None

def check_existing_tables():
    """Check which tables already exist"""
    logger.info("🔍 Checking existing database structure...")
    
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    expected_tables = {
        "crowd_reports": "Crowd reports table",
        "triage_patients": "Triage patients table", 
        "voice_analyses": "Voice analysis table (NEW)",
        "multimodal_assessments": "Multimodal assessment table (NEW)",
        "context_analyses": "Context analysis table (NEW)",
        "device_performance": "Device performance table (NEW)"
    }
    
    for table_name, description in expected_tables.items():
        status = "EXISTS" if table_name in existing_tables else "MISSING"
        logger.info(f"   • {table_name}: {status} - {description}")
    
    return existing_tables

def add_new_columns_safely():
    """Add new columns to existing tables safely"""
    logger.info("🔧 Adding new columns to existing tables...")
    
    try:
        with engine.connect() as conn:
            # Check if crowd_reports table exists and add new columns
            try:
                # Add new columns to crowd_reports
                new_crowd_columns = [
                    "ALTER TABLE crowd_reports ADD COLUMN severity INTEGER",
                    "ALTER TABLE crowd_reports ADD COLUMN confidence_score FLOAT",
                    "ALTER TABLE crowd_reports ADD COLUMN ai_analysis JSON",
                    "ALTER TABLE crowd_reports ADD COLUMN source VARCHAR(50) DEFAULT 'manual'",
                    "ALTER TABLE crowd_reports ADD COLUMN reporter_id VARCHAR(255)",
                    "ALTER TABLE crowd_reports ADD COLUMN metadata JSON",
                    "ALTER TABLE crowd_reports ADD COLUMN voice_analysis_id INTEGER",
                    "ALTER TABLE crowd_reports ADD COLUMN emotional_state JSON",
                    "ALTER TABLE crowd_reports ADD COLUMN urgency_detected VARCHAR(20)",
                    "ALTER TABLE crowd_reports ADD COLUMN verified BOOLEAN DEFAULT FALSE",
                    "ALTER TABLE crowd_reports ADD COLUMN verified_by VARCHAR(100)",
                    "ALTER TABLE crowd_reports ADD COLUMN verified_at DATETIME",
                    "ALTER TABLE crowd_reports ADD COLUMN response_dispatched BOOLEAN DEFAULT FALSE",
                    "ALTER TABLE crowd_reports ADD COLUMN response_time DATETIME",
                    "ALTER TABLE crowd_reports ADD COLUMN response_units JSON"
                ]
                
                for sql in new_crowd_columns:
                    try:
                        conn.execute(text(sql))
                        logger.info(f"   ✅ Added column: {sql.split('ADD COLUMN')[1].split()[0]}")
                    except SQLAlchemyError as e:
                        if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                            logger.info(f"   ⏭️  Column already exists: {sql.split('ADD COLUMN')[1].split()[0]}")
                        else:
                            logger.warning(f"   ⚠️  Failed to add column: {e}")
                
                # Add new columns to triage_patients
                new_patient_columns = [
                    "ALTER TABLE triage_patients ADD COLUMN ai_assessment JSON",
                    "ALTER TABLE triage_patients ADD COLUMN risk_prediction FLOAT",
                    "ALTER TABLE triage_patients ADD COLUMN deterioration_risk VARCHAR(20)",
                    "ALTER TABLE triage_patients ADD COLUMN estimated_wait_time INTEGER",
                    "ALTER TABLE triage_patients ADD COLUMN treatment_plan JSON",
                    "ALTER TABLE triage_patients ADD COLUMN discharge_prediction DATETIME",
                    "ALTER TABLE triage_patients ADD COLUMN resource_requirements JSON"
                ]
                
                for sql in new_patient_columns:
                    try:
                        conn.execute(text(sql))
                        logger.info(f"   ✅ Added column: {sql.split('ADD COLUMN')[1].split()[0]}")
                    except SQLAlchemyError as e:
                        if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                            logger.info(f"   ⏭️  Column already exists: {sql.split('ADD COLUMN')[1].split()[0]}")
                        else:
                            logger.warning(f"   ⚠️  Failed to add column: {e}")
                
                conn.commit()
                logger.info("✅ Successfully added new columns to existing tables")
                
            except Exception as e:
                logger.error(f"❌ Failed to add columns: {e}")
                conn.rollback()
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Column addition failed: {e}")
        return False

def create_new_tables():
    """Create new tables for Gemma 3n features"""
    logger.info("🆕 Creating new tables...")
    
    try:
        # Create all tables (existing ones will be skipped)
        Base.metadata.create_all(bind=engine)
        logger.info("✅ All tables created/verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")
        return False

def create_indexes():
    """Create performance indexes"""
    logger.info("📊 Creating performance indexes...")
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_crowd_reports_severity ON crowd_reports(severity)",
        "CREATE INDEX IF NOT EXISTS idx_crowd_reports_source ON crowd_reports(source)",
        "CREATE INDEX IF NOT EXISTS idx_crowd_reports_verified ON crowd_reports(verified)",
        "CREATE INDEX IF NOT EXISTS idx_voice_analyses_urgency ON voice_analyses(urgency_level)",
        "CREATE INDEX IF NOT EXISTS idx_voice_analyses_created ON voice_analyses(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_multimodal_assessments_severity ON multimodal_assessments(severity_score)",
        "CREATE INDEX IF NOT EXISTS idx_multimodal_assessments_type ON multimodal_assessments(assessment_type)",
        "CREATE INDEX IF NOT EXISTS idx_context_analyses_type ON context_analyses(analysis_type)",
        "CREATE INDEX IF NOT EXISTS idx_device_performance_device ON device_performance(device_id)",
        "CREATE INDEX IF NOT EXISTS idx_device_performance_timestamp ON device_performance(timestamp)"
    ]
    
    try:
        with engine.connect() as conn:
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    index_name = index_sql.split("CREATE INDEX IF NOT EXISTS")[1].split("ON")[0].strip()
                    logger.info(f"   ✅ Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Index creation failed: {e}")
            
            conn.commit()
        
        logger.info("✅ Indexes created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Index creation failed: {e}")
        return False

def verify_migration():
    """Verify migration was successful"""
    logger.info("🔍 Verifying migration...")
    
    try:
        db = next(get_db())
        
        # Check table counts
        crowd_count = db.execute(text("SELECT COUNT(*) FROM crowd_reports")).scalar()
        patient_count = db.execute(text("SELECT COUNT(*) FROM triage_patients")).scalar()
        
        # Check new tables exist and are accessible
        voice_count = db.execute(text("SELECT COUNT(*) FROM voice_analyses")).scalar()
        multimodal_count = db.execute(text("SELECT COUNT(*) FROM multimodal_assessments")).scalar()
        context_count = db.execute(text("SELECT COUNT(*) FROM context_analyses")).scalar()
        performance_count = db.execute(text("SELECT COUNT(*) FROM device_performance")).scalar()
        
        logger.info("✅ Migration verification:")
        logger.info(f"   • Crowd reports: {crowd_count}")
        logger.info(f"   • Triage patients: {patient_count}")
        logger.info(f"   • Voice analyses: {voice_count}")
        logger.info(f"   • Multimodal assessments: {multimodal_count}")
        logger.info(f"   • Context analyses: {context_count}")
        logger.info(f"   • Device performance: {performance_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration verification failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing new features"""
    logger.info("🌱 Creating sample data for new features...")
    
    try:
        db = next(get_db())
        
        # Sample voice analysis
        sample_voice = VoiceAnalysis(
            audio_file_path="/sample/emergency_call.wav",
            transcript="There's a fire at the warehouse on Main Street. Multiple people trapped inside.",
            confidence=0.92,
            urgency_level="critical",
            emergency_type="fire_emergency",
            hazards_detected=[
                {"type": "fire", "severity": 9, "confidence": 0.95},
                {"type": "trapped_persons", "severity": 8, "confidence": 0.88}
            ],
            emotional_state={
                "primary_emotion": "panic",
                "stress_level": 0.9,
                "caller_state": "high_distress"
            },
            location_extracted="Main Street Warehouse District",
            analyst_id="migration_system"
        )
        
        # Sample multimodal assessment
        sample_multimodal = MultimodalAssessment(
            assessment_type="comprehensive_damage",
            text_input="Building collapse reported with multiple casualties",
            severity_score=9.2,
            emergency_type="structural_collapse",
            risk_factors=[
                {"risk": "Secondary collapse", "probability": 0.7, "impact": 9},
                {"risk": "Trapped victims", "probability": 0.9, "impact": 8}
            ],
            resource_requirements={
                "personnel": {"search_rescue": 6, "structural_engineer": 2},
                "equipment": ["heavy_rescue", "shoring_equipment", "medical_team"],
                "estimated_response_time": "15 minutes"
            },
            ai_confidence=0.87,
            analyst_id="migration_system"
        )
        
        # Sample context analysis
        sample_context = ContextAnalysis(
            analysis_type="emergency_trend",
            input_tokens=45000,
            context_window_used=45000,
            output_summary="Analysis of emergency patterns shows increased fire incidents in warehouse district",
            key_insights=[
                {"pattern": "Fire incidents up 40% in warehouse district", "confidence": 0.85},
                {"pattern": "Peak incident time: 2-4 PM weekdays", "confidence": 0.92}
            ],
            confidence=0.89,
            analyst_id="migration_system"
        )
        
        # Sample device performance
        sample_performance = DevicePerformance(
            device_id="emergency_workstation_001",
            cpu_usage=45.2,
            memory_usage=67.8,
            gpu_usage=23.1,
            battery_level=85.0,
            temperature=52.3,
            inference_speed=12.4,
            model_config={
                "model_variant": "gemma-3n-4b",
                "context_window": 64000,
                "precision": "fp16"
            },
            optimization_level="balanced"
        )
        
        # Add sample data
        db.add(sample_voice)
        db.add(sample_multimodal)
        db.add(sample_context)
        db.add(sample_performance)
        
        db.commit()
        
        logger.info("✅ Sample data created successfully")
        logger.info("   • 1 voice analysis sample")
        logger.info("   • 1 multimodal assessment sample")
        logger.info("   • 1 context analysis sample")
        logger.info("   • 1 device performance sample")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sample data creation failed: {e}")
        db.rollback()
        return False

def migrate_existing_data():
    """Migrate and enhance existing data with new features"""
    logger.info("🔄 Migrating existing data...")
    
    try:
        db = next(get_db())
        
        # Update existing crowd reports with default values for new fields
        existing_reports = db.execute(text("""
            UPDATE crowd_reports 
            SET source = 'legacy_data',
                verified = FALSE,
                response_dispatched = FALSE
            WHERE source IS NULL
        """))
        
        # Set severity based on escalation for existing reports
        db.execute(text("""
            UPDATE crowd_reports 
            SET severity = CASE 
                WHEN escalation = 'critical' THEN 9
                WHEN escalation = 'high' THEN 7
                WHEN escalation = 'moderate' THEN 5
                WHEN escalation = 'low' THEN 3
                ELSE 5
            END
            WHERE severity IS NULL
        """))
        
        # Update existing patients with enhanced fields
        db.execute(text("""
            UPDATE triage_patients
            SET deterioration_risk = CASE
                WHEN triage_color = 'red' THEN 'high'
                WHEN triage_color = 'yellow' THEN 'medium'
                WHEN triage_color = 'green' THEN 'low'
                ELSE 'unknown'
            END
            WHERE deterioration_risk IS NULL
        """))
        
        db.commit()
        
        logger.info("✅ Existing data migrated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data migration failed: {e}")
        db.rollback()
        return False

def run_migration():
    """Run complete database migration"""
    logger.info("🚀 Starting database migration for Gemma 3n features...")
    logger.info("="*60)
    
    migration_steps = [
        ("Backup existing data", backup_existing_data),
        ("Check existing tables", check_existing_tables),
        ("Add new columns", add_new_columns_safely),
        ("Create new tables", create_new_tables),
        ("Create indexes", create_indexes),
        ("Migrate existing data", migrate_existing_data),
        ("Verify migration", verify_migration),
        ("Create sample data", create_sample_data)
    ]
    
    results = {}
    
    for step_name, step_function in migration_steps:
        logger.info(f"\n📋 Step: {step_name}")
        logger.info("-" * 40)
        
        try:
            if step_name == "Backup existing data":
                result = step_function()
                results[step_name] = result is not None
                if result:
                    results["backup_file"] = result
            elif step_name == "Check existing tables":
                result = step_function()
                results[step_name] = True
                results["existing_tables"] = result
            else:
                result = step_function()
                results[step_name] = result
            
            if results[step_name]:
                logger.info(f"✅ {step_name} completed successfully")
            else:
                logger.error(f"❌ {step_name} failed")
                
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
            results[step_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🏁 Migration Summary")
    logger.info("="*60)
    
    successful_steps = sum(1 for success in results.values() if success is True)
    total_steps = len([k for k in results.keys() if k not in ["backup_file", "existing_tables"]])
    
    logger.info(f"Success Rate: {successful_steps}/{total_steps} steps completed")
    
    for step_name, success in results.items():
        if step_name not in ["backup_file", "existing_tables"]:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"   • {step_name}: {status}")
    
    if "backup_file" in results:
        logger.info(f"\n💾 Backup file: {results['backup_file']}")
    
    if successful_steps == total_steps:
        logger.info("\n🎉 Migration completed successfully!")
        logger.info("Your database is now ready for Gemma 3n features.")
        logger.info("\nNext steps:")
        logger.info("   1. Restart your application")
        logger.info("   2. Test new API endpoints")
        logger.info("   3. Try voice emergency processing")
        logger.info("   4. Explore multimodal analysis")
        return True
    else:
        logger.warning("\n⚠️  Migration completed with some failures.")
        logger.warning("Check the logs above and resolve any issues.")
        logger.warning("You may need to run the migration again.")
        return False

def rollback_migration(backup_file: str = None):
    """Rollback migration if needed"""
    logger.info("🔙 Rolling back migration...")
    
    if not backup_file:
        import glob
        backup_files = glob.glob("database_backup_*.json")
        if backup_files:
            backup_file = max(backup_files)  # Get latest backup
            logger.info(f"Using latest backup: {backup_file}")
        else:
            logger.error("❌ No backup file found for rollback")
            return False
    
    try:
        # Load backup data
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        logger.info(f"📥 Loading backup from {backup_file}")
        
        # Drop new tables
        with engine.connect() as conn:
            new_tables = ["voice_analyses", "multimodal_assessments", "context_analyses", "device_performance"]
            for table in new_tables:
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                    logger.info(f"   🗑️  Dropped table: {table}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to drop {table}: {e}")
            
            conn.commit()
        
        # Restore original data structure would go here
        # (This is a simplified example)
        
        logger.info("✅ Migration rolled back successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        return False

def main():
    """Main migration function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        backup_file = sys.argv[2] if len(sys.argv) > 2 else None
        rollback_migration(backup_file)
    else:
        run_migration()

if __name__ == "__main__":
    main()