# app/celery_app.py - Celery configuration for background tasks

from celery import Celery
from celery.schedules import crontab
import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
BROKER_URL = REDIS_URL
RESULT_BACKEND = REDIS_URL

# Create Celery app
celery_app = Celery(
    'disaster_response',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        'app.tasks.voice_processing',
        'app.tasks.image_analysis',
        'app.tasks.report_generation',
        'app.tasks.monitoring',
        'app.tasks.notifications'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'app.tasks.voice_processing.*': {'queue': 'voice'},
        'app.tasks.image_analysis.*': {'queue': 'image'},
        'app.tasks.report_generation.*': {'queue': 'reports'},
        'app.tasks.monitoring.*': {'queue': 'monitoring'},
        'app.tasks.notifications.*': {'queue': 'notifications'},
    },
    
    # Task execution
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_default_queue='default',
    task_default_exchange='default',
    task_default_exchange_type='direct',
    task_default_routing_key='default',
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Task timeouts
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    
    # Result backend settings
    result_expires=3600,       # 1 hour
    result_persistent=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        # System monitoring every 5 minutes
        'monitor-system-performance': {
            'task': 'app.tasks.monitoring.monitor_system_performance',
            'schedule': crontab(minute='*/5'),
            'options': {'queue': 'monitoring'}
        },
        
        # Check for urgent reports every minute
        'check-urgent-reports': {
            'task': 'app.tasks.monitoring.check_urgent_reports',
            'schedule': crontab(minute='*'),
            'options': {'queue': 'monitoring'}
        },
        
        # Cleanup old files daily at 2 AM
        'cleanup-old-files': {
            'task': 'app.tasks.monitoring.cleanup_old_files',
            'schedule': crontab(hour=2, minute=0),
            'options': {'queue': 'monitoring'}
        },
        
        # Generate daily summary report at 6 AM
        'daily-summary-report': {
            'task': 'app.tasks.report_generation.generate_daily_summary',
            'schedule': crontab(hour=6, minute=0),
            'options': {'queue': 'reports'}
        },
        
        # Backup database daily at 3 AM
        'backup-database': {
            'task': 'app.tasks.monitoring.backup_database',
            'schedule': crontab(hour=3, minute=0),
            'options': {'queue': 'monitoring'}
        },
        
        # AI model performance check every hour
        'check-ai-performance': {
            'task': 'app.tasks.monitoring.check_ai_performance',
            'schedule': crontab(minute=0),
            'options': {'queue': 'monitoring'}
        }
    },
    
    # Queue configuration
    task_queues={
        'voice': {
            'exchange': 'voice',
            'exchange_type': 'direct',
            'routing_key': 'voice',
        },
        'image': {
            'exchange': 'image', 
            'exchange_type': 'direct',
            'routing_key': 'image',
        },
        'reports': {
            'exchange': 'reports',
            'exchange_type': 'direct',
            'routing_key': 'reports',
        },
        'monitoring': {
            'exchange': 'monitoring',
            'exchange_type': 'direct',
            'routing_key': 'monitoring',
        },
        'notifications': {
            'exchange': 'notifications',
            'exchange_type': 'direct',
            'routing_key': 'notifications',
        }
    }
)

# Task retry configuration
def task_retry_config(max_retries=3, countdown=60):
    """Standard retry configuration for tasks"""
    return {
        'max_retries': max_retries,
        'default_retry_delay': countdown,
        'retry_backoff': True,
        'retry_backoff_max': 600,  # 10 minutes max
        'retry_jitter': True
    }

# Priority levels for emergency tasks
PRIORITY_CRITICAL = 9
PRIORITY_HIGH = 7
PRIORITY_MEDIUM = 5
PRIORITY_LOW = 3
PRIORITY_ROUTINE = 1

def get_task_priority(urgency_level: str) -> int:
    """Get Celery task priority based on urgency level"""
    priority_map = {
        'critical': PRIORITY_CRITICAL,
        'high': PRIORITY_HIGH,
        'medium': PRIORITY_MEDIUM,
        'low': PRIORITY_LOW,
        'routine': PRIORITY_ROUTINE
    }
    return priority_map.get(urgency_level.lower(), PRIORITY_MEDIUM)

# Task decorators for common emergency response patterns
def emergency_task(bind=True, **kwargs):
    """Decorator for emergency processing tasks"""
    defaults = {
        'bind': bind,
        'autoretry_for': (Exception,),
        **task_retry_config(),
        **kwargs
    }
    return celery_app.task(**defaults)

def monitoring_task(bind=True, **kwargs):
    """Decorator for monitoring tasks"""
    defaults = {
        'bind': bind,
        'queue': 'monitoring',
        'autoretry_for': (Exception,),
        **task_retry_config(max_retries=1),
        **kwargs
    }
    return celery_app.task(**defaults)

def report_task(bind=True, **kwargs):
    """Decorator for report generation tasks"""
    defaults = {
        'bind': bind,
        'queue': 'reports',
        'autoretry_for': (Exception,),
        **task_retry_config(),
        **kwargs
    }
    return celery_app.task(**defaults)

# Health check for Celery
@celery_app.task(bind=True)
def health_check(self):
    """Health check task for Celery workers"""
    try:
        from app.database import get_db
        from app.adaptive_ai_settings import get_current_performance
        
        # Check database connection
        db = next(get_db())
        db.close()
        
        # Check AI system
        performance = get_current_performance()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'worker_id': self.request.hostname,
            'task_id': self.request.id,
            'performance': {
                'cpu_usage': performance.cpu_usage,
                'memory_usage': performance.memory_usage,
                'inference_speed': performance.inference_speed
            }
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat(),
            'worker_id': self.request.hostname,
            'task_id': self.request.id
        }

# Startup tasks
@celery_app.task
def initialize_ai_models():
    """Initialize AI models on worker startup"""
    try:
        from app.inference import Gemma3nEmergencyProcessor
        from app.audio_transcription import VoiceEmergencyProcessor
        from app.adaptive_ai_settings import optimize_for_emergency
        
        logger.info("Initializing AI models on worker...")
        
        # Initialize Gemma processor
        processor = Gemma3nEmergencyProcessor()
        logger.info("Gemma 3n processor initialized")
        
        # Initialize voice processor
        voice_processor = VoiceEmergencyProcessor()
        logger.info("Voice processor initialized")
        
        # Optimize for emergency use case
        config = optimize_for_emergency()
        logger.info(f"AI optimized: {config.model_variant}")
        
        return {
            'status': 'initialized',
            'timestamp': datetime.utcnow().isoformat(),
            'model_config': {
                'model_variant': config.model_variant,
                'context_window': config.context_window,
                'precision': config.precision
            }
        }
        
    except Exception as e:
        logger.error(f"AI model initialization failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

# Worker startup signal
@celery_app.task(bind=True)
def worker_startup_tasks(self):
    """Tasks to run when worker starts"""
    logger.info(f"Worker {self.request.hostname} starting up...")
    
    # Initialize AI models
    init_result = initialize_ai_models.delay()
    
    # Run health check
    health_result = health_check.delay()
    
    return {
        'worker_id': self.request.hostname,
        'startup_time': datetime.utcnow().isoformat(),
        'initialization_task': init_result.id,
        'health_check_task': health_result.id
    }

# Graceful shutdown
@celery_app.task
def worker_shutdown_cleanup():
    """Cleanup tasks before worker shutdown"""
    logger.info("Worker shutting down, performing cleanup...")
    
    try:
        # Close database connections
        from app.database import engine
        engine.dispose()
        
        # Clear AI model cache if needed
        import gc
        gc.collect()
        
        return {
            'status': 'cleanup_completed',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Shutdown cleanup failed: {e}")
        return {
            'status': 'cleanup_failed',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

# Custom task base class for emergency response
class EmergencyTask(celery_app.Task):
    """Base class for emergency response tasks with enhanced error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"Emergency task {task_id} failed: {exc}")
        
        # Send notification for critical task failures
        if kwargs.get('urgency_level') == 'critical':
            from app.tasks.notifications import send_critical_alert
            send_critical_alert.delay(
                subject=f"Critical Task Failure: {task_id}",
                message=f"Emergency task failed with error: {exc}",
                task_id=task_id
            )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        if kwargs.get('urgency_level') == 'critical':
            logger.info(f"Critical emergency task {task_id} completed successfully")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Emergency task {task_id} retrying due to: {exc}")

# Register custom task base
celery_app.Task = EmergencyTask

if __name__ == '__main__':
    celery_app.start()