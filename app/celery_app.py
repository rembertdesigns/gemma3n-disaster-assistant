# -*- coding: utf-8 -*-
"""
celery_app.py - Celery Task Queue for the Disaster Response Assistant.

This module configures and defines all asynchronous background tasks for the application,
including AI processing, system monitoring, report generation, and notifications.
It uses Celery with Redis as the broker and result backend. It includes a fallback
mock for development environments where Celery/Redis may not be available.
"""

import os
import sys
import logging
import asyncio
import gc
import shutil
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path to allow for clean imports of app modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System monitoring imports with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ psutil not installed. System monitoring will be limited.")
    PSUTIL_AVAILABLE = False

# Celery imports with a robust fallback mock for local development without Redis/Celery
try:
    from celery import Celery
    from celery.schedules import crontab
    from celery.signals import worker_ready, worker_shutdown
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
    logger.info("✅ Celery imported successfully. Asynchronous tasks are enabled.")
except ImportError:
    logger.warning("⚠️ Celery not installed. Background tasks will run synchronously as mock functions.")
    CELERY_AVAILABLE = False
    
    # Create a comprehensive mock Celery for development to avoid crashes
    class MockTaskResult:
        def __init__(self, result):
            self.result = result
            self.id = f"mock-task-{uuid.uuid4()}"
            self.status = 'SUCCESS'
        def ready(self): return True
        def failed(self): return False
        @property
        def traceback(self): return None

    class MockCelery:
        def __init__(self, *args, **kwargs):
            self.conf = type('MockConf', (), {'update': lambda *a, **kw: None})()
            self.Task = type('MockTask', (), {}) # Base class for tasks
        
        def task(self, *args, **kwargs):
            def decorator(func):
                def delay(*a, **kw):
                    logger.info(f"Executing mock task '{func.__name__}' synchronously.")
                    return MockTaskResult(func(None, *a, **kw)) # Pass None for bound 'self'
                func.delay = delay
                func.apply_async = lambda args, **kw: delay(*args)
                return func
            return decorator
        
        def start(self):
            logger.error("Cannot start Celery worker because Celery is not installed.")
        
        @property
        def control(self):
            return type('MockControl', (), {'inspect': lambda: None})()

# ================================================================================
# CELERY CONFIGURATION
# ================================================================================

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

if CELERY_AVAILABLE:
    celery_app = Celery(
        'celery_app',  # Name of the current module
        broker=BROKER_URL,
        backend=RESULT_BACKEND,
        include=['celery_app']
    )
else:
    celery_app = MockCelery('celery_app')

if CELERY_AVAILABLE:
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True, # Acknowledges task after it completes, not when it starts
        worker_max_tasks_per_child=100, # Prevents memory leaks
        task_soft_time_limit=300, # 5 minutes soft limit
        task_time_limit=600, # 10 minutes hard limit
        result_expires=timedelta(hours=1),
        result_persistent=True,
        task_reject_on_worker_lost=True,
        
        # Task routes with correct 'celery_app.' prefix
        task_routes={
            'celery_app.process_voice_emergency': {'queue': 'voice'},
            'celery_app.analyze_image_hazards': {'queue': 'image'},
            'celery_app.process_multimodal_emergency': {'queue': 'ai_processing'},
            'celery_app.send_emergency_notification': {'queue': 'notifications'},
            'celery_app.monitor_system_performance': {'queue': 'monitoring'},
            'celery_app.generate_hourly_summary': {'queue': 'reports'},
            'celery_app.health_check': {'queue': 'monitoring'},
        },
        
        # Beat scheduler for periodic tasks
        beat_schedule={
            'monitor-system-critical': {
                'task': 'celery_app.monitor_system_performance',
                'schedule': crontab(minute='*/2'),
                'options': {'queue': 'monitoring'}
            },
            'check-urgent-reports': {
                'task': 'celery_app.check_urgent_reports',
                'schedule': crontab(minute='*'),
                'options': {'queue': 'monitoring'}
            },
            'monitor-ai-performance': {
                'task': 'celery_app.monitor_ai_performance',
                'schedule': crontab(minute='*/5'),
                'options': {'queue': 'monitoring'}
            },
            'cleanup-temp-files': {
                'task': 'celery_app.cleanup_temp_files',
                'schedule': crontab(minute='*/30'),
                'options': {'queue': 'monitoring'}
            },
            'generate-hourly-summary': {
                'task': 'celery_app.generate_hourly_summary',
                'schedule': crontab(hour='*'),
                'options': {'queue': 'reports'}
            },
            'daily-backup': {
                'task': 'celery_app.backup_database',
                'schedule': crontab(hour=3, minute=0),
                'options': {'queue': 'monitoring'}
            },
            'monthly-data-cleanup': {
                'task': 'celery_app.monthly_data_cleanup',
                'schedule': crontab(hour=2, minute=0, day_of_month=1),
                'options': {'queue': 'monitoring'}
            },
            'worker-health-check': {
                'task': 'celery_app.health_check',
                'schedule': crontab(minute='*/10'),
                'options': {'queue': 'monitoring'}
            }
        }
    )

# ================================================================================
# TASK DECORATORS & UTILS
# ================================================================================

def emergency_task(**kwargs):
    """Decorator for high-priority emergency tasks with standard retry logic."""
    defaults = {
        'bind': True,
        'autoretry_for': (Exception,),
        'retry_kwargs': {'max_retries': 3, 'countdown': 60},
        'task_acks_late': True,
        **kwargs
    }
    return celery_app.task(**defaults)

def get_system_performance():
    """Get current system performance metrics"""
    if not PSUTIL_AVAILABLE:
        return {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'error': 'psutil not available'
        }
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent,
            'memory_available': memory.available,
            'disk_free': disk.free
        }
    except Exception as e:
        logger.error(f"Failed to get system performance: {e}")
        return {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'error': str(e)
        }

# ================================================================================
# EMERGENCY PROCESSING TASKS
# ================================================================================

@emergency_task(queue='voice')
def process_voice_emergency(self, audio_file_path: str, urgency_level: str = 'medium'):
    """Processes a voice recording to analyze for emergency content."""
    logger.info(f"Task {self.request.id}: Processing voice emergency from {audio_file_path}")
    
    try:
        # Dynamic imports to avoid circular dependencies at module level
        from api import voice_processor, VoiceAnalysis, get_db
        
        analysis = voice_processor.process_emergency_call(audio_file_path)
        
        db = next(get_db())
        record = VoiceAnalysis(
            audio_file_path=audio_file_path,
            transcript=analysis["transcript"],
            confidence=analysis["confidence"],
            urgency_level=analysis["overall_urgency"],
            emotional_state=analysis["emotional_state"],
            hazards_detected=analysis["hazards_detected"],
            analyst_id="celery_worker"
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        # Send critical alert if needed
        if analysis["overall_urgency"] == 'critical':
            send_emergency_notification.delay(
                notification_type='critical_voice_analysis', 
                data=analysis
            )
            
        logger.info(f"Voice emergency processed successfully: {record.id}")
        return {
            'status': 'success', 
            'analysis_id': record.id,
            'urgency_level': analysis["overall_urgency"],
            'confidence': analysis["confidence"]
        }
        
    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {e}", exc_info=True)
        raise self.retry(exc=e)
    finally:
        if 'db' in locals():
            db.close()

@emergency_task(queue='image')
def analyze_image_hazards(self, image_path: str, urgency_level: str = 'medium'):
    """Analyzes an image to detect hazards."""
    logger.info(f"Task {self.request.id}: Analyzing image {image_path}")
    
    try:
        # Dynamic import from our single API file
        from api import detect_hazards
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        hazards = detect_hazards(image_data)
        
        # Create analysis result
        analysis_result = {
            'hazards_detected': hazards,
            'severity_score': len(hazards) * 2.5,
            'confidence': 0.85,
            'image_path': image_path,
            'processing_time': datetime.utcnow().isoformat()
        }
        
        # Send critical alert if dangerous hazards detected
        if 'fire' in hazards or len(hazards) > 2:
            send_emergency_notification.delay(
                notification_type='critical_image_hazards', 
                data=analysis_result
            )
            
        logger.info(f"Image analysis completed: {len(hazards)} hazards detected")
        return {
            'status': 'success', 
            'hazards': hazards,
            'analysis': analysis_result
        }
        
    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {e}", exc_info=True)
        raise self.retry(exc=e)

@emergency_task(queue='ai_processing')
def process_multimodal_emergency(self, text_input: str = None, image_path: str = None, 
                                audio_path: str = None, urgency_level: str = 'medium'):
    """Process multimodal emergency input using AI analysis."""
    logger.info(f"Task {self.request.id}: Processing multimodal emergency analysis")
    
    try:
        # Dynamic imports from our API
        from api import gemma_processor, MultimodalAssessment, get_db
        
        # Read file data if provided
        image_data = None
        audio_data = None
        
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image_data = f.read()
                
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
        
        # Process with Gemma 3n processor
        analysis_result = gemma_processor.analyze_multimodal_emergency(
            text=text_input,
            image_data=image_data,
            audio_data=audio_data
        )
        
        # Save to database
        db = next(get_db())
        assessment = MultimodalAssessment(
            assessment_type="celery_multimodal",
            text_input=text_input,
            image_path=image_path,
            audio_path=audio_path,
            severity_score=analysis_result["severity"]["overall_score"],
            emergency_type=analysis_result["emergency_type"]["primary"],
            risk_factors=analysis_result["immediate_risks"],
            resource_requirements=analysis_result["resource_requirements"],
            ai_confidence=analysis_result["severity"]["confidence"],
            analyst_id="celery_worker"
        )
        
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        
        # Critical emergency response
        if analysis_result["severity"]["overall_score"] >= 8.0:
            send_emergency_notification.delay(
                notification_type='critical_multimodal_emergency',
                data=analysis_result
            )
        
        logger.info(f"Multimodal emergency processed: severity {analysis_result['severity']['overall_score']}")
        return {
            'status': 'success',
            'assessment_id': assessment.id,
            'analysis_result': analysis_result,
            'processing_time': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {e}", exc_info=True)
        raise self.retry(exc=e)
    finally:
        if 'db' in locals():
            db.close()

# ================================================================================
# MONITORING & MAINTENANCE TASKS
# ================================================================================

@celery_app.task(bind=True, name="celery_app.monitor_system_performance")
def monitor_system_performance(self):
    """Monitors system vitals like CPU, memory, and disk."""
    logger.info("Running system performance monitoring task...")
    
    try:
        performance = get_system_performance()
        
        # Check for critical conditions and send alerts
        alerts = []
        if performance.get('cpu_usage', 0) > 90:
            alerts.append(f"High CPU usage: {performance['cpu_usage']:.1f}%")
        if performance.get('memory_usage', 0) > 90:
            alerts.append(f"High memory usage: {performance['memory_usage']:.1f}%")
        if performance.get('disk_usage', 0) > 90:
            alerts.append(f"High disk usage: {performance['disk_usage']:.1f}%")
        
        if alerts:
            logger.warning(f"High system resource usage detected: {performance}")
            send_emergency_notification.delay(
                notification_type='system_performance_alert',
                data={'alerts': alerts, 'performance': performance}
            )
        
        # Store performance data if possible
        try:
            from api import DevicePerformance, get_db
            db = next(get_db())
            
            perf_record = DevicePerformance(
                device_id="main_server",
                cpu_usage=performance.get('cpu_usage', 0),
                memory_usage=performance.get('memory_usage', 0),
                gpu_usage=0,  # Placeholder
                battery_level=100,  # Placeholder for server
                inference_speed=0.2,  # Placeholder
                temperature=25.0  # Placeholder
            )
            
            db.add(perf_record)
            db.commit()
            db.close()
        except Exception as db_error:
            logger.warning(f"Could not store performance data: {db_error}")
        
        return performance
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}", exc_info=True)
        return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

@celery_app.task(bind=True, name="celery_app.check_urgent_reports")
def check_urgent_reports(self):
    """Periodically checks the database for unresolved urgent reports."""
    logger.info("Checking for urgent pending reports...")
    
    try:
        from api import EmergencyReport, CrowdReport, get_db
        
        db = next(get_db())
        five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
        
        # Check critical emergency reports
        critical_emergency_reports = db.query(EmergencyReport).filter(
            EmergencyReport.priority == 'critical',
            EmergencyReport.status == 'pending',
            EmergencyReport.timestamp < five_minutes_ago
        ).count()
        
        # Check critical crowd reports
        critical_crowd_reports = db.query(CrowdReport).filter(
            CrowdReport.escalation == 'critical',
            CrowdReport.status == 'pending',
            CrowdReport.timestamp < five_minutes_ago
        ).count()
        
        total_urgent = critical_emergency_reports + critical_crowd_reports
        
        if total_urgent > 0:
            logger.critical(f"{total_urgent} critical reports are pending for over 5 minutes!")
            send_emergency_notification.delay(
                notification_type='urgent_reports_pending',
                data={
                    'total_urgent': total_urgent,
                    'emergency_reports': critical_emergency_reports,
                    'crowd_reports': critical_crowd_reports,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
        
        return {
            'pending_critical_reports': total_urgent,
            'emergency_reports': critical_emergency_reports,
            'crowd_reports': critical_crowd_reports,
            'check_time': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Urgent report check failed: {e}", exc_info=True)
        return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    finally:
        if 'db' in locals():
            db.close()

@celery_app.task(bind=True, name="celery_app.monitor_ai_performance")
def monitor_ai_performance(self):
    """Monitors the performance of AI components."""
    logger.info("Monitoring AI component performance...")
    
    try:
        from api import ai_optimizer
        
        performance = ai_optimizer.monitor_performance()
        
        # Check for performance issues
        performance_issues = []
        if performance.cpu_usage > 85:
            performance_issues.append("High CPU usage affecting AI processing")
        if performance.memory_usage > 85:
            performance_issues.append("High memory usage may impact AI models")
        if performance.inference_speed > 2.0:
            performance_issues.append("Slow AI inference detected")
        
        # Auto-optimize if needed
        if performance_issues:
            logger.warning(f"AI performance issues detected: {performance_issues}")
            new_config = ai_optimizer.optimize_for_device("emergency")
            ai_optimizer.current_config = new_config
            
            send_emergency_notification.delay(
                notification_type='ai_performance_optimization',
                data={
                    'issues': performance_issues,
                    'optimization_applied': True,
                    'new_config': {
                        'model_variant': new_config.model_variant,
                        'context_window': new_config.context_window,
                        'optimization_level': new_config.optimization_level
                    }
                }
            )
        
        return {
            'status': 'healthy' if not performance_issues else 'optimized',
            'performance': {
                'cpu_usage': performance.cpu_usage,
                'memory_usage': performance.memory_usage,
                'inference_speed': performance.inference_speed,
                'temperature': performance.temperature
            },
            'issues_detected': len(performance_issues),
            'optimization_applied': len(performance_issues) > 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI performance monitoring failed: {e}", exc_info=True)
        return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

@celery_app.task(bind=True, name="celery_app.cleanup_temp_files")
def cleanup_temp_files(self):
    """Cleans up old files from the temporary upload directory."""
    logger.info("Running temporary file cleanup task...")
    
    try:
        from api import UPLOAD_DIR
        
        count = 0
        one_hour_ago = time.time() - 3600
        
        for temp_file in UPLOAD_DIR.glob("temp_*"):
            try:
                if temp_file.is_file() and temp_file.stat().st_mtime < one_hour_ago:
                    temp_file.unlink()
                    count += 1
            except OSError as e:
                logger.warning(f"Could not delete temp file {temp_file}: {e}")
        
        logger.info(f"Cleaned up {count} old temporary files.")
        return {
            'files_cleaned': count,
            'cleanup_time': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}", exc_info=True)
        return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

@celery_app.task(bind=True, name="celery_app.health_check")
def health_check(self):
    """Comprehensive health check for Celery workers."""
    logger.info("Running Celery worker health check...")
    
    try:
        # Test database connection
        from api import get_db
        db = next(get_db())
        db.close()
        db_status = "healthy"
        
        # Test AI components
        from api import gemma_processor, voice_processor, ai_optimizer
        ai_status = "healthy"
        
        # Get system performance
        performance = get_system_performance()
        
        # Get worker info
        worker_id = getattr(self.request, 'hostname', 'unknown')
        task_id = getattr(self.request, 'id', 'unknown')
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'worker_id': worker_id,
            'task_id': task_id,
            'components': {
                'database': db_status,
                'ai_system': ai_status,
                'system_performance': performance
            },
            'celery_info': {
                'worker_available': True,
                'task_processing': True
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat(),
            'worker_id': getattr(self.request, 'hostname', 'unknown'),
            'task_id': getattr(self.request, 'id', 'unknown')
        }

# ================================================================================
# REPORTING & NOTIFICATION TASKS
# ================================================================================

@celery_app.task(bind=True, name="celery_app.generate_hourly_summary")
def generate_hourly_summary(self):
    """Generates an hourly summary of all incidents."""
    logger.info("Generating hourly incident summary report...")
    
    try:
        from api import CrowdReport, EmergencyReport, TriagePatient, get_db
        
        db = next(get_db())
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        # Gather statistics
        crowd_reports = db.query(CrowdReport).filter(
            CrowdReport.timestamp >= one_hour_ago
        ).count()
        
        emergency_reports = db.query(EmergencyReport).filter(
            EmergencyReport.timestamp >= one_hour_ago
        ).count()
        
        new_patients = db.query(TriagePatient).filter(
            TriagePatient.created_at >= one_hour_ago
        ).count()
        
        critical_reports = db.query(CrowdReport).filter(
            CrowdReport.timestamp >= one_hour_ago,
            CrowdReport.escalation == 'critical'
        ).count()
        
        summary = {
            'period': 'hourly',
            'timestamp': datetime.utcnow().isoformat(),
            'statistics': {
                'crowd_reports': crowd_reports,
                'emergency_reports': emergency_reports,
                'new_patients': new_patients,
                'critical_reports': critical_reports,
                'total_incidents': crowd_reports + emergency_reports
            }
        }
        
        # Send summary if there's significant activity
        if summary['statistics']['total_incidents'] > 5 or critical_reports > 0:
            send_emergency_notification.delay(
                notification_type='hourly_activity_summary',
                data=summary
            )
        
        logger.info(f"Hourly summary: {summary['statistics']['total_incidents']} total incidents")
        return summary
        
    except Exception as e:
        logger.error(f"Hourly summary generation failed: {e}", exc_info=True)
        return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    finally:
        if 'db' in locals():
            db.close()

@celery_app.task(bind=True, name="celery_app.backup_database")
def backup_database(self):
    """Performs a daily backup of the application database."""
    logger.info("Starting daily database backup...")
    
    try:
        from api import BASE_DIR, config
        
        # Determine database path
        db_path = None
        if hasattr(config, 'DATABASE_URL') and 'sqlite' in config.DATABASE_URL:
            db_path = Path(config.DATABASE_URL.split("///")[-1])
        else:
            # Fallback to common SQLite location
            db_path = BASE_DIR / "data" / "emergency_response.db"
        
        if not db_path.exists():
            logger.warning(f"Database file not found at {db_path}")
            return {'status': 'warning', 'message': 'Database file not found'}
        
        # Create backup directory
        backup_dir = BASE_DIR / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"emergency_db_backup_{timestamp}.db"
        
        # Copy database file
        shutil.copy2(db_path, backup_path)
        
        # Keep only last 7 backups
        backups = sorted(backup_dir.glob("emergency_db_backup_*.db"))
        while len(backups) > 7:
            oldest_backup = backups.pop(0)
            oldest_backup.unlink()
            logger.info(f"Removed old backup: {oldest_backup.name}")
        
        logger.info(f"Successfully created database backup at {backup_path}")
        return {
            'status': 'success',
            'backup_file': backup_path.name,
            'backup_size': backup_path.stat().st_size,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}", exc_info=True)
        return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

@celery_app.task(bind=True, name="celery_app.monthly_data_cleanup")
def monthly_data_cleanup(self):
    """Performs a monthly cleanup of old, resolved records."""
    logger.info("Starting monthly data cleanup...")
    
    try:
        from api import CrowdReport, EmergencyReport, VoiceAnalysis, DevicePerformance, get_db
        
        db = next(get_db())
        
        # Keep data for 90 days
        cleanup_date = datetime.utcnow() - timedelta(days=90)
        
        # Clean old crowd reports
        old_crowd_reports = db.query(CrowdReport).filter(
            CrowdReport.timestamp < cleanup_date
        ).count()
        db.query(CrowdReport).filter(CrowdReport.timestamp < cleanup_date).delete()
        
        # Clean old voice analyses
        old_voice_analyses = db.query(VoiceAnalysis).filter(
            VoiceAnalysis.created_at < cleanup_date
        ).count()
        db.query(VoiceAnalysis).filter(VoiceAnalysis.created_at < cleanup_date).delete()
        
        # Clean old performance records (keep last 30 days only)
        perf_cleanup_date = datetime.utcnow() - timedelta(days=30)
        old_performance = db.query(DevicePerformance).filter(
            DevicePerformance.timestamp < perf_cleanup_date
        ).count()
        db.query(DevicePerformance).filter(DevicePerformance.timestamp < perf_cleanup_date).delete()
        
        db.commit()
        
        cleanup_summary = {
            'period': 'monthly',
            'cleanup_date': cleanup_date.isoformat(),
            'records_cleaned': {
                'crowd_reports': old_crowd_reports,
                'voice_analyses': old_voice_analyses,
                'performance_records': old_performance
            },
            'total_cleaned': old_crowd_reports + old_voice_analyses + old_performance,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Monthly cleanup completed: {cleanup_summary['total_cleaned']} records removed")
        return cleanup_summary
        
    except Exception as e:
        logger.error(f"Monthly cleanup failed: {e}", exc_info=True)
        return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    finally:
        if 'db' in locals():
            db.close()

@celery_app.task(bind=True, name="celery_app.send_emergency_notification")
def send_emergency_notification(self, notification_type: str, data: Dict[str, Any]):
    """Sends a high-priority notification to relevant parties."""
    logger.info(f"Sending notification of type '{notification_type}': {data}")
    
    try:
        notification_data = {
            'type': notification_type,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        # Determine priority level
        critical_types = [
            'critical_voice_analysis',
            'critical_image_hazards', 
            'critical_multimodal_emergency',
            'system_performance_alert',
            'urgent_reports_pending'
        ]
        
        is_critical = notification_type in critical_types
        
        # In a real implementation, this would send to:
        # - Email alerts (SMTP)
        # - SMS notifications (Twilio)
        # - Push notifications (Firebase)
        # - Webhook endpoints
        # - Emergency service APIs
        # - Slack/Teams channels
        
        # For now, log appropriately based on priority
        if is_critical:
            logger.critical(f"🚨 CRITICAL NOTIFICATION: {notification_type}")
            logger.critical(f"Data: {json.dumps(data, indent=2)}")
        else:
            logger.info(f"📢 Notification sent: {notification_type}")
            logger.info(f"Data: {json.dumps(data, indent=2)}")
        
        # Simulate notification sending
        notification_result = {
            'status': 'sent',
            'notification_type': notification_type,
            'priority': 'critical' if is_critical else 'normal',
            'channels': ['log'],  # In real implementation: ['email', 'sms', 'push']
            'timestamp': datetime.utcnow().isoformat(),
            'recipients': 1  # In real implementation: actual recipient count
        }
        
        return notification_result
        
    except Exception as e:
        logger.error(f"Failed to send notification '{notification_type}': {e}", exc_info=True)
        raise self.retry(exc=e, countdown=30, max_retries=3)

# ================================================================================
# UTILITY FUNCTIONS FOR TASK MANAGEMENT
# ================================================================================

def schedule_emergency_analysis(audio_path: str = None, image_path: str = None, 
                               text_input: str = None, urgency_level: str = 'medium'):
    """Schedule emergency analysis tasks based on available inputs."""
    
    if not CELERY_AVAILABLE:
        logger.warning("Celery not available - running tasks synchronously")
        results = {}
        
        if audio_path:
            results['voice'] = process_voice_emergency(None, audio_path, urgency_level)
        if image_path:
            results['image'] = analyze_image_hazards(None, image_path, urgency_level)
        if text_input or image_path or audio_path:
            results['multimodal'] = process_multimodal_emergency(
                None, text_input, image_path, audio_path, urgency_level
            )
        
        return results
    
    # Schedule tasks with Celery
    scheduled_tasks = {}
    
    try:
        if audio_path:
            task = process_voice_emergency.delay(audio_path, urgency_level)
            scheduled_tasks['voice_analysis'] = task.id
            logger.info(f"Scheduled voice analysis task: {task.id}")
            
        if image_path:
            task = analyze_image_hazards.delay(image_path, urgency_level)
            scheduled_tasks['image_analysis'] = task.id
            logger.info(f"Scheduled image analysis task: {task.id}")
            
        if text_input or image_path or audio_path:
            task = process_multimodal_emergency.delay(text_input, image_path, audio_path, urgency_level)
            scheduled_tasks['multimodal_analysis'] = task.id
            logger.info(f"Scheduled multimodal analysis task: {task.id}")
        
        logger.info(f"Scheduled {len(scheduled_tasks)} emergency analysis tasks")
        return scheduled_tasks
        
    except Exception as e:
        logger.error(f"Failed to schedule emergency tasks: {e}")
        return {'error': str(e)}

def get_task_status(task_id: str):
    """Get status of a specific task."""
    if not CELERY_AVAILABLE:
        return {'status': 'mock', 'message': 'Celery not available'}
    
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        return {
            'task_id': task_id,
            'status': result.status,
            'result': result.result if result.ready() else None,
            'traceback': result.traceback if result.failed() else None,
            'ready': result.ready()
        }
    except Exception as e:
        return {'error': str(e)}

def get_worker_stats():
    """Get statistics about active workers."""
    if not CELERY_AVAILABLE:
        return {'workers': 0, 'message': 'Celery not available'}
    
    try:
        inspect = celery_app.control.inspect()
        
        active_workers = inspect.active()
        scheduled_tasks = inspect.scheduled()
        worker_stats = inspect.stats()
        
        return {
            'active_workers': len(active_workers) if active_workers else 0,
            'total_active_tasks': sum(
                len(tasks) for tasks in active_workers.values()
            ) if active_workers else 0,
            'scheduled_tasks': sum(
                len(tasks) for tasks in scheduled_tasks.values()
            ) if scheduled_tasks else 0,
            'worker_details': worker_stats
        }
    except Exception as e:
        return {'error': str(e)}

def test_all_tasks():
    """Test all task types for development/debugging."""
    if not CELERY_AVAILABLE:
        logger.warning("Celery not available - cannot run task tests")
        return False
    
    try:
        logger.info("🧪 Testing all emergency tasks...")
        
        # Test health check
        health_task = health_check.delay()
        logger.info(f"✅ Health check scheduled: {health_task.id}")
        
        # Test system monitoring
        monitor_task = monitor_system_performance.delay()
        logger.info(f"✅ System monitoring scheduled: {monitor_task.id}")
        
        # Test AI monitoring
        ai_monitor_task = monitor_ai_performance.delay()
        logger.info(f"✅ AI monitoring scheduled: {ai_monitor_task.id}")
        
        # Test notification
        notify_task = send_emergency_notification.delay(
            notification_type='test_notification',
            data={'message': 'This is a test notification from Celery'}
        )
        logger.info(f"✅ Test notification scheduled: {notify_task.id}")
        
        # Test temp file cleanup
        cleanup_task = cleanup_temp_files.delay()
        logger.info(f"✅ Cleanup task scheduled: {cleanup_task.id}")
        
        logger.info("🎉 All test tasks scheduled successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Task testing failed: {e}")
        return False

# ================================================================================
# CELERY SIGNALS & WORKER LIFECYCLE
# ================================================================================

if CELERY_AVAILABLE:
    @worker_ready.connect
    def worker_ready_handler(sender=None, **kwargs):
        """Handler for when a Celery worker is ready and connected."""
        logger.info(f"🚀 Celery worker '{sender.hostname}' is ready and connected to the broker.")
        
        # Run initial health check
        health_check.delay()
        
        # Log worker capabilities
        logger.info("Worker capabilities:")
        logger.info("  • Emergency voice processing")
        logger.info("  • Image hazard analysis") 
        logger.info("  • Multimodal AI analysis")
        logger.info("  • System performance monitoring")
        logger.info("  • Emergency notifications")
        logger.info("  • Automated maintenance tasks")

    @worker_shutdown.connect
    def worker_shutdown_handler(sender=None, **kwargs):
        """Handler for when a Celery worker is shutting down."""
        logger.info(f"🛑 Celery worker '{sender.hostname}' is shutting down.")
        
        try:
            # Cleanup resources
            gc.collect()
            
            # Close database connections if possible
            try:
                from api import engine
                engine.dispose()
                logger.info("Database connections closed")
            except:
                pass
            
        except Exception as e:
            logger.error(f"Worker shutdown cleanup failed: {e}")

# ================================================================================
# DEVELOPMENT & DEBUGGING UTILITIES
# ================================================================================

def run_development_tasks():
    """Run a series of tasks for development testing."""
    logger.info("🔧 Running development task suite...")
    
    # Test basic functionality
    results = {}
    
    try:
        # Test system monitoring
        results['system_monitor'] = monitor_system_performance.delay()
        
        # Test AI performance monitoring
        results['ai_monitor'] = monitor_ai_performance.delay()
        
        # Test cleanup
        results['cleanup'] = cleanup_temp_files.delay()
        
        # Test notification
        results['notification'] = send_emergency_notification.delay(
            'development_test',
            {'message': 'Development test notification', 'timestamp': datetime.utcnow().isoformat()}
        )
        
        # Test health check
        results['health'] = health_check.delay()
        
        logger.info(f"🎯 Scheduled {len(results)} development tasks")
        
        # Wait a moment and check results
        import time
        time.sleep(2)
        
        for task_name, task_result in results.items():
            status = get_task_status(task_result.id)
            logger.info(f"  • {task_name}: {status['status']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Development task suite failed: {e}")
        return {'error': str(e)}

def get_celery_info():
    """Get comprehensive information about Celery setup."""
    info = {
        'celery_available': CELERY_AVAILABLE,
        'psutil_available': PSUTIL_AVAILABLE,
        'broker_url': BROKER_URL if CELERY_AVAILABLE else 'N/A',
        'result_backend': RESULT_BACKEND if CELERY_AVAILABLE else 'N/A',
        'task_queues': [
            'voice', 'image', 'ai_processing', 
            'notifications', 'monitoring', 'reports'
        ] if CELERY_AVAILABLE else [],
        'scheduled_tasks': [
            'monitor-system-critical',
            'check-urgent-reports', 
            'monitor-ai-performance',
            'cleanup-temp-files',
            'generate-hourly-summary',
            'daily-backup',
            'monthly-data-cleanup',
            'worker-health-check'
        ] if CELERY_AVAILABLE else []
    }
    
    if CELERY_AVAILABLE:
        try:
            worker_stats = get_worker_stats()
            info.update(worker_stats)
        except:
            info['worker_stats_error'] = 'Could not retrieve worker stats'
    
    return info

# ================================================================================
# CLI ENTRY POINT
# ================================================================================

def main():
    """Main entry point for running Celery worker."""
    if not CELERY_AVAILABLE:
        logger.error("❌ Celery is not installed. Please run 'pip install celery redis' to use background workers.")
        logger.info("💡 Alternative: Run tasks synchronously by calling them directly from your API")
        sys.exit(1)
    
    logger.info("🚀 Starting Enhanced Emergency Response Celery Worker")
    logger.info("📋 Configuration:")
    logger.info(f"   • Broker: {BROKER_URL}")
    logger.info(f"   • Backend: {RESULT_BACKEND}")
    logger.info(f"   • System monitoring: {'✅' if PSUTIL_AVAILABLE else '❌'}")
    logger.info("📡 Available queues:")
    logger.info("   • voice - Voice emergency processing")
    logger.info("   • image - Image hazard analysis")
    logger.info("   • ai_processing - Multimodal AI analysis")
    logger.info("   • notifications - Emergency alerts")
    logger.info("   • monitoring - System monitoring")
    logger.info("   • reports - Report generation")
    logger.info("")
    logger.info("🏃 Starting Celery worker...")
    logger.info("💡 Use Ctrl+C to stop the worker")
    
    # Start Celery worker
    try:
        celery_app.start()
    except KeyboardInterrupt:
        logger.info("👋 Celery worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Celery worker failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()