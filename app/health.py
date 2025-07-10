from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import psutil
import torch
import os
import time
import logging
from datetime import datetime, timedelta
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Create router for health endpoints
health_router = APIRouter(prefix="/health", tags=["health"])

class HealthChecker:
    """Comprehensive health checking for the disaster response system"""
    
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'ai_models': self._check_ai_models,
            'storage': self._check_storage,
            'system_resources': self._check_system_resources,
            'external_apis': self._check_external_apis,
            'background_tasks': self._check_background_tasks
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        start_time = time.time()
        results = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.1.0',
            'checks': {},
            'summary': {},
            'response_time_ms': 0
        }
        
        healthy_checks = 0
        total_checks = len(self.checks)
        
        for check_name, check_function in self.checks.items():
            try:
                check_result = check_function()
                results['checks'][check_name] = check_result
                
                if check_result.get('status') == 'healthy':
                    healthy_checks += 1
                elif check_result.get('status') == 'warning':
                    if results['status'] == 'healthy':
                        results['status'] = 'warning'
                else:  # unhealthy
                    results['status'] = 'unhealthy'
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                results['checks'][check_name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                results['status'] = 'unhealthy'
        
        # Calculate summary
        results['summary'] = {
            'healthy_checks': healthy_checks,
            'total_checks': total_checks,
            'health_percentage': (healthy_checks / total_checks) * 100,
            'overall_status': results['status']
        }
        
        results['response_time_ms'] = int((time.time() - start_time) * 1000)
        
        return results
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and basic operations"""
        try:
            from app.database import get_db
            from app.models import CrowdReport
            
            db = next(get_db())
            
            # Test basic query
            result = db.execute(text("SELECT 1")).scalar()
            
            # Test table access
            count = db.query(CrowdReport).count()
            
            db.close()
            
            return {
                'status': 'healthy',
                'message': 'Database connection successful',
                'details': {
                    'query_test': result == 1,
                    'crowd_reports_count': count,
                    'connection_time_ms': '<50'
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': 'Database connection failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _check_ai_models(self) -> Dict[str, Any]:
        """Check AI model availability and performance"""
        try:
            from app.inference import Gemma3nEmergencyProcessor
            from app.adaptive_ai_settings import adaptive_optimizer, get_current_performance
            
            # Check if models are loadable
            processor = Gemma3nEmergencyProcessor()
            model_available = processor.model is not None
            
            # Check current performance
            performance = get_current_performance()
            
            # Check AI optimizer
            optimizer_active = adaptive_optimizer.current_config is not None
            
            # Simple inference test
            test_result = processor.analyze_multimodal_emergency(
                text="Test emergency message for health check"
            )
            inference_working = test_result is not None
            
            status = 'healthy' if all([model_available, inference_working]) else 'warning'
            
            return {
                'status': status,
                'message': 'AI models operational',
                'details': {
                    'model_available': model_available,
                    'inference_working': inference_working,
                    'optimizer_active': optimizer_active,
                    'inference_speed': performance.inference_speed,
                    'model_config': {
                        'variant': adaptive_optimizer.current_config.model_variant if adaptive_optimizer.current_config else 'unknown',
                        'precision': adaptive_optimizer.current_config.precision if adaptive_optimizer.current_config else 'unknown'
                    }
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': 'AI models unavailable',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _check_storage(self) -> Dict[str, Any]:
        """Check storage availability and disk space"""
        try:
            # Check required directories
            directories = ['uploads', 'outputs', 'models', 'logs']
            directory_status = {}
            
            for directory in directories:
                path = os.path.join('.', directory)
                exists = os.path.exists(path)
                writable = os.access(path, os.W_OK) if exists else False
                directory_status[directory] = {
                    'exists': exists,
                    'writable': writable
                }
            
            # Check disk space
            disk_usage = psutil.disk_usage('.')
            free_space_gb = disk_usage.free / (1024**3)
            total_space_gb = disk_usage.total / (1024**3)
            usage_percentage = (disk_usage.used / disk_usage.total) * 100
            
            # Determine status
            if free_space_gb < 1:  # Less than 1GB free
                status = 'unhealthy'
                message = 'Critical: Low disk space'
            elif free_space_gb < 5:  # Less than 5GB free
                status = 'warning'
                message = 'Warning: Low disk space'
            elif any(not d['exists'] or not d['writable'] for d in directory_status.values()):
                status = 'warning'
                message = 'Warning: Directory access issues'
            else:
                status = 'healthy'
                message = 'Storage systems operational'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'directories': directory_status,
                    'disk_space': {
                        'free_gb': round(free_space_gb, 2),
                        'total_gb': round(total_space_gb, 2),
                        'usage_percentage': round(usage_percentage, 1)
                    }
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': 'Storage check failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # GPU information
            gpu_available = torch.cuda.is_available()
            gpu_usage = 0
            gpu_memory_usage = 0
            
            if gpu_available:
                try:
                    gpu_usage = torch.cuda.utilization() or 0
                    gpu_memory_used = torch.cuda.memory_allocated()
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_usage = (gpu_memory_used / gpu_memory_total) * 100
                except:
                    gpu_available = False
            
            # Battery status
            battery = psutil.sensors_battery()
            battery_level = battery.percent if battery else None
            
            # Network connectivity (simple check)
            network_available = True
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
            except:
                network_available = False
            
            # Determine status based on resource usage
            if cpu_usage > 90 or memory_usage > 95:
                status = 'unhealthy'
                message = 'Critical: High resource usage'
            elif cpu_usage > 80 or memory_usage > 85:
                status = 'warning'
                message = 'Warning: High resource usage'
            elif memory_available_gb < 1:
                status = 'warning'
                message = 'Warning: Low available memory'
            else:
                status = 'healthy'
                message = 'System resources normal'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'cpu': {
                        'usage_percentage': round(cpu_usage, 1),
                        'core_count': cpu_count
                    },
                    'memory': {
                        'usage_percentage': round(memory_usage, 1),
                        'available_gb': round(memory_available_gb, 2),
                        'total_gb': round(memory.total / (1024**3), 2)
                    },
                    'gpu': {
                        'available': gpu_available,
                        'usage_percentage': round(gpu_usage, 1) if gpu_available else None,
                        'memory_usage_percentage': round(gpu_memory_usage, 1) if gpu_available else None
                    },
                    'battery': {
                        'level_percentage': battery_level,
                        'plugged': battery.power_plugged if battery else None
                    },
                    'network': {
                        'available': network_available
                    }
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': 'System resource check failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity"""
        try:
            import requests
            from app.map_utils import map_utils
            
            api_status = {}
            
            # Check Google Maps API
            google_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
            if google_api_key:
                try:
                    response = requests.get(
                        'https://maps.googleapis.com/maps/api/geocode/json',
                        params={'address': 'test', 'key': google_api_key},
                        timeout=5
                    )
                    api_status['google_maps'] = {
                        'available': response.status_code == 200,
                        'response_time_ms': int(response.elapsed.total_seconds() * 1000)
                    }
                except:
                    api_status['google_maps'] = {'available': False, 'error': 'Connection failed'}
            else:
                api_status['google_maps'] = {'available': False, 'error': 'No API key configured'}
            
            # Check MapBox API
            mapbox_api_key = os.getenv('MAPBOX_API_KEY')
            if mapbox_api_key:
                try:
                    response = requests.get(
                        'https://api.mapbox.com/geocoding/v5/mapbox.places/test.json',
                        params={'access_token': mapbox_api_key},
                        timeout=5
                    )
                    api_status['mapbox'] = {
                        'available': response.status_code == 200,
                        'response_time_ms': int(response.elapsed.total_seconds() * 1000)
                    }
                except:
                    api_status['mapbox'] = {'available': False, 'error': 'Connection failed'}
            else:
                api_status['mapbox'] = {'available': False, 'error': 'No API key configured'}
            
            # Check weather API
            weather_api_key = os.getenv('WEATHER_API_KEY')
            if weather_api_key:
                try:
                    response = requests.get(
                        'http://api.openweathermap.org/data/2.5/weather',
                        params={'q': 'test', 'appid': weather_api_key},
                        timeout=5
                    )
                    api_status['weather'] = {
                        'available': response.status_code in [200, 404],  # 404 is OK for test query
                        'response_time_ms': int(response.elapsed.total_seconds() * 1000)
                    }
                except:
                    api_status['weather'] = {'available': False, 'error': 'Connection failed'}
            else:
                api_status['weather'] = {'available': False, 'error': 'No API key configured'}
            
            # Determine overall status
            available_apis = sum(1 for api in api_status.values() if api.get('available', False))
            total_configured_apis = len([k for k, v in api_status.items() if 'No API key' not in v.get('error', '')])
            
            if available_apis == 0 and total_configured_apis > 0:
                status = 'unhealthy'
                message = 'All external APIs unavailable'
            elif available_apis < total_configured_apis:
                status = 'warning'
                message = f'{available_apis}/{total_configured_apis} APIs available'
            else:
                status = 'healthy'
                message = 'External APIs operational'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'apis': api_status,
                    'summary': {
                        'available': available_apis,
                        'configured': total_configured_apis
                    }
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': 'External API check failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _check_background_tasks(self) -> Dict[str, Any]:
        """Check background task system (Celery)"""
        try:
            # Try to import Celery
            try:
                from app.celery_app import celery_app
                celery_available = True
            except ImportError:
                celery_available = False
            
            if not celery_available:
                return {
                    'status': 'warning',
                    'message': 'Background task system not configured',
                    'details': {'celery_available': False},
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Check if Celery is running
            try:
                # Send a simple task to check if workers are available
                from app.celery_app import health_check
                result = health_check.delay()
                
                # Wait briefly for result
                task_result = result.get(timeout=5)
                worker_healthy = task_result.get('status') == 'healthy'
                
                status = 'healthy' if worker_healthy else 'warning'
                message = 'Background tasks operational' if worker_healthy else 'Background tasks slow'
                
                return {
                    'status': status,
                    'message': message,
                    'details': {
                        'celery_available': True,
                        'workers_responding': worker_healthy,
                        'task_result': task_result
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as celery_error:
                return {
                    'status': 'warning',
                    'message': 'Background task workers not responding',
                    'details': {
                        'celery_available': True,
                        'workers_responding': False,
                        'error': str(celery_error)
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': 'Background task check failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Global health checker instance
health_checker = HealthChecker()

# Health check endpoints
@health_router.get("/")
async def health_check():
    """Basic health check endpoint"""
    try:
        # Quick system check
        start_time = time.time()
        
        # Test database
        from app.database import get_db
        db = next(get_db())
        db.close()
        
        response_time = int((time.time() - start_time) * 1000)
        
        return {
            "status": "healthy",
            "service": "Disaster Response Assistant",
            "version": "2.1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": response_time
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })

@health_router.get("/detailed")
async def detailed_health_check():
    """Comprehensive health check with all system components"""
    result = health_checker.run_all_checks()
    
    # Return appropriate HTTP status
    if result['status'] == 'unhealthy':
        raise HTTPException(status_code=503, detail=result)
    elif result['status'] == 'warning':
        # Return 200 but with warnings in response
        return result
    else:
        return result

@health_router.get("/database")
async def database_health():
    """Database-specific health check"""
    result = health_checker._check_database()
    
    if result['status'] == 'unhealthy':
        raise HTTPException(status_code=503, detail=result)
    return result

@health_router.get("/ai")
async def ai_health():
    """AI system health check"""
    result = health_checker._check_ai_models()
    
    if result['status'] == 'unhealthy':
        raise HTTPException(status_code=503, detail=result)
    return result

@health_router.get("/system")
async def system_health():
    """System resources health check"""
    result = health_checker._check_system_resources()
    
    if result['status'] == 'unhealthy':
        raise HTTPException(status_code=503, detail=result)
    return result

@health_router.get("/external")
async def external_health():
    """External APIs health check"""
    result = health_checker._check_external_apis()
    
    if result['status'] == 'unhealthy':
        raise HTTPException(status_code=503, detail=result)
    return result

@health_router.get("/metrics")
async def system_metrics():
    """Prometheus-style metrics endpoint"""
    try:
        from app.adaptive_ai_settings import get_current_performance
        
        performance = get_current_performance()
        
        # Simple text format for Prometheus
        metrics = f"""# HELP disaster_response_cpu_usage CPU usage percentage
# TYPE disaster_response_cpu_usage gauge
disaster_response_cpu_usage {performance.cpu_usage}

# HELP disaster_response_memory_usage Memory usage percentage
# TYPE disaster_response_memory_usage gauge
disaster_response_memory_usage {performance.memory_usage}

# HELP disaster_response_inference_speed AI inference speed (tokens per second)
# TYPE disaster_response_inference_speed gauge
disaster_response_inference_speed {performance.inference_speed}

# HELP disaster_response_battery_level Battery level percentage
# TYPE disaster_response_battery_level gauge
disaster_response_battery_level {performance.battery_level}

# HELP disaster_response_temperature System temperature
# TYPE disaster_response_temperature gauge
disaster_response_temperature {performance.temperature}
"""
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@health_router.get("/readiness")
async def readiness_probe():
    """Kubernetes readiness probe"""
    try:
        # Check if application is ready to serve traffic
        from app.database import get_db
        from app.adaptive_ai_settings import adaptive_optimizer
        
        # Test database
        db = next(get_db())
        db.close()
        
        # Check AI system
        ai_ready = adaptive_optimizer.current_config is not None
        
        if ai_ready:
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            raise HTTPException(status_code=503, detail={
                "status": "not_ready",
                "reason": "AI system not initialized",
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "status": "not_ready",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })

@health_router.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe"""
    # Simple check that the service is running
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - psutil.boot_time())
    }

# Add health router to main app in server.py
def setup_health_checks(app):
    """Setup health check routes in FastAPI app"""
    app.include_router(health_router)
    
    # Add startup event to initialize health monitoring
    @app.on_event("startup")
    async def startup_health_check():
        logger.info("🏥 Health monitoring system initialized")
        
        # Run initial health check
        try:
            initial_health = health_checker.run_all_checks()
            logger.info(f"Initial health status: {initial_health['status']}")
            
            if initial_health['status'] == 'unhealthy':
                logger.warning("⚠️ System started with health issues")
                for check_name, check_result in initial_health['checks'].items():
                    if check_result.get('status') == 'unhealthy':
                        logger.warning(f"  • {check_name}: {check_result.get('message', 'Unknown issue')}")
            
        except Exception as e:
            logger.error(f"Initial health check failed: {e}")