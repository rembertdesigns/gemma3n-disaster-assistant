# app/adaptive_ai_settings.py - Adaptive AI Optimization for Gemma 3n

import psutil
import torch
import platform
import subprocess
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import threading
import time
import os

logger = logging.getLogger(__name__)

@dataclass
class DeviceCapabilities:
    """Device hardware capabilities"""
    cpu_cores: int
    cpu_frequency: float  # GHz
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    gpu_name: str
    battery_level: float
    network_speed: str
    storage_free_gb: float
    platform: str
    architecture: str

@dataclass 
class GemmaConfig:
    """Gemma 3n model configuration"""
    model_variant: str  # 3n-2b, 3n-4b, 3n-4b-hq
    context_window: int
    batch_size: int
    precision: str  # fp16, fp32, int8, int4
    optimization_level: str  # speed, balanced, quality
    max_tokens: int
    temperature: float
    use_flash_attention: bool
    gradient_checkpointing: bool
    cpu_offload: bool

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory_usage: float
    battery_level: float
    temperature: float
    inference_speed: float  # tokens per second
    power_consumption: float  # watts
    network_latency: float  # ms
    disk_io: float
    timestamp: datetime

class AdaptiveAIOptimizer:
    """Adaptive AI optimization system for Gemma 3n"""
    
    def __init__(self):
        self.device_caps = self._detect_device_capabilities()
        self.current_config = None
        self.performance_history = []
        self.optimization_rules = self._load_optimization_rules()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            "cpu_critical": 90.0,
            "cpu_high": 80.0,
            "memory_critical": 95.0,
            "memory_high": 85.0,
            "battery_low": 20.0,
            "battery_critical": 10.0,
            "temperature_high": 80.0,
            "temperature_critical": 90.0,
            "inference_slow": 5.0,  # tokens per second
            "network_slow": 1000.0  # ms latency
        }
        
        logger.info(f"Adaptive AI Optimizer initialized for {self.device_caps.platform}")
        logger.info(f"Device: {self.device_caps.cpu_cores} cores, {self.device_caps.memory_gb:.1f}GB RAM, GPU: {self.device_caps.gpu_available}")
    
    def _detect_device_capabilities(self) -> DeviceCapabilities:
        """Detect comprehensive device capabilities"""
        
        try:
            # CPU information
            cpu_count = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.current / 1000 if cpu_freq else 2.0  # Convert to GHz
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # GPU information
            gpu_available = torch.cuda.is_available()
            gpu_memory_gb = 0.0
            gpu_name = "None"
            
            if gpu_available:
                try:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_name = torch.cuda.get_device_name(0)
                except:
                    gpu_available = False
            
            # Battery information
            battery = psutil.sensors_battery()
            battery_level = battery.percent if battery else 100.0
            
            # Storage information
            disk = psutil.disk_usage('/')
            storage_free_gb = disk.free / (1024**3)
            
            # Platform information
            platform_name = platform.system()
            architecture = platform.machine()
            
            # Network speed estimation
            network_speed = self._estimate_network_speed()
            
            return DeviceCapabilities(
                cpu_cores=cpu_count,
                cpu_frequency=cpu_frequency,
                memory_gb=memory_gb,
                gpu_available=gpu_available,
                gpu_memory_gb=gpu_memory_gb,
                gpu_name=gpu_name,
                battery_level=battery_level,
                network_speed=network_speed,
                storage_free_gb=storage_free_gb,
                platform=platform_name,
                architecture=architecture
            )
            
        except Exception as e:
            logger.error(f"Error detecting device capabilities: {e}")
            # Return minimal capabilities as fallback
            return DeviceCapabilities(
                cpu_cores=4, cpu_frequency=2.0, memory_gb=8.0,
                gpu_available=False, gpu_memory_gb=0.0, gpu_name="Unknown",
                battery_level=100.0, network_speed="unknown",
                storage_free_gb=10.0, platform="Unknown", architecture="unknown"
            )
    
    def _estimate_network_speed(self) -> str:
        """Estimate network connection speed"""
        try:
            # Simple ping test to estimate network quality
            import subprocess
            result = subprocess.run(['ping', '-c', '3', '8.8.8.8'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'avg' in line:
                        # Extract average ping time
                        avg_time = float(line.split('/')[-3])
                        if avg_time < 50:
                            return "fast"
                        elif avg_time < 200:
                            return "medium"
                        else:
                            return "slow"
            
            return "unknown"
            
        except:
            return "unknown"
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules for different scenarios"""
        
        return {
            "emergency": {
                "priority": "speed",
                "max_cpu_usage": 95.0,
                "max_memory_usage": 90.0,
                "min_inference_speed": 10.0,
                "power_saving": False
            },
            "routine": {
                "priority": "balanced",
                "max_cpu_usage": 70.0,
                "max_memory_usage": 80.0,
                "min_inference_speed": 5.0,
                "power_saving": True
            },
            "battery_saver": {
                "priority": "efficiency",
                "max_cpu_usage": 50.0,
                "max_memory_usage": 70.0,
                "min_inference_speed": 3.0,
                "power_saving": True
            }
        }
    
    def optimize_for_device(self, use_case: str = "emergency") -> GemmaConfig:
        """Optimize Gemma 3n configuration for current device and use case"""
        
        logger.info(f"Optimizing for use case: {use_case}")
        
        # Get optimization rules for use case
        rules = self.optimization_rules.get(use_case, self.optimization_rules["emergency"])
        
        # Base configuration selection
        if self.device_caps.memory_gb < 4:
            base_config = self._get_low_resource_config()
        elif self.device_caps.memory_gb < 8:
            base_config = self._get_balanced_config()
        elif self.device_caps.memory_gb < 16:
            base_config = self._get_high_performance_config()
        else:
            base_config = self._get_maximum_performance_config()
        
        # Apply use case specific optimizations
        optimized_config = self._apply_optimization_rules(base_config, rules)
        
        # Apply device-specific optimizations
        final_config = self._apply_device_optimizations(optimized_config)
        
        self.current_config = final_config
        logger.info(f"Optimized config: {final_config.model_variant}, {final_config.context_window} context, {final_config.precision} precision")
        
        return final_config
    
    def _get_low_resource_config(self) -> GemmaConfig:
        """Configuration for low-resource devices"""
        return GemmaConfig(
            model_variant="gemma-3n-2b",
            context_window=16000,
            batch_size=1,
            precision="int8",
            optimization_level="speed",
            max_tokens=256,
            temperature=0.3,
            use_flash_attention=False,
            gradient_checkpointing=True,
            cpu_offload=True
        )
    
    def _get_balanced_config(self) -> GemmaConfig:
        """Configuration for balanced performance"""
        return GemmaConfig(
            model_variant="gemma-3n-4b",
            context_window=32000,
            batch_size=2,
            precision="fp16",
            optimization_level="balanced",
            max_tokens=512,
            temperature=0.3,
            use_flash_attention=True,
            gradient_checkpointing=True,
            cpu_offload=False
        )
    
    def _get_high_performance_config(self) -> GemmaConfig:
        """Configuration for high-performance devices"""
        return GemmaConfig(
            model_variant="gemma-3n-4b",
            context_window=64000,
            batch_size=4,
            precision="fp16",
            optimization_level="quality",
            max_tokens=1024,
            temperature=0.3,
            use_flash_attention=True,
            gradient_checkpointing=False,
            cpu_offload=False
        )
    
    def _get_maximum_performance_config(self) -> GemmaConfig:
        """Configuration for maximum performance devices"""
        return GemmaConfig(
            model_variant="gemma-3n-4b-hq",
            context_window=128000,
            batch_size=8,
            precision="fp16",
            optimization_level="quality",
            max_tokens=2048,
            temperature=0.3,
            use_flash_attention=True,
            gradient_checkpointing=False,
            cpu_offload=False
        )
    
    def _apply_optimization_rules(self, config: GemmaConfig, rules: Dict[str, Any]) -> GemmaConfig:
        """Apply optimization rules to configuration"""
        
        # Adjust based on priority
        if rules["priority"] == "speed":
            config.precision = "int8" if config.precision == "fp16" else config.precision
            config.context_window = min(config.context_window, 32000)
            config.use_flash_attention = True
        elif rules["priority"] == "efficiency":
            config.precision = "int8"
            config.context_window = min(config.context_window, 16000)
            config.batch_size = 1
            config.cpu_offload = True
        
        # Apply power saving if needed
        if rules["power_saving"] and self.device_caps.battery_level < 50:
            config = self._apply_power_saving(config)
        
        return config
    
    def _apply_device_optimizations(self, config: GemmaConfig) -> GemmaConfig:
        """Apply device-specific optimizations"""
        
        # GPU optimizations
        if not self.device_caps.gpu_available:
            config.cpu_offload = True
            config.precision = "int8"  # Better for CPU inference
            config.batch_size = 1
        elif self.device_caps.gpu_memory_gb < 4:
            config.precision = "int8"
            config.batch_size = min(config.batch_size, 2)
            config.gradient_checkpointing = True
        
        # Memory optimizations
        memory_usage_estimate = self._estimate_memory_usage(config)
        available_memory = self.device_caps.memory_gb * 0.8  # Leave 20% for system
        
        if memory_usage_estimate > available_memory:
            # Reduce memory usage
            config.context_window = int(config.context_window * 0.7)
            config.batch_size = max(1, config.batch_size // 2)
            config.gradient_checkpointing = True
        
        # CPU optimizations
        if self.device_caps.cpu_cores < 4:
            config.batch_size = 1
            config.cpu_offload = True
        
        # Platform-specific optimizations
        if self.device_caps.platform == "Darwin":  # macOS
            config.use_flash_attention = False  # May not be optimized
        elif self.device_caps.platform == "Windows":
            # Windows-specific optimizations
            pass
        
        return config
    
    def _apply_power_saving(self, config: GemmaConfig) -> GemmaConfig:
        """Apply power saving optimizations"""
        
        config.model_variant = "gemma-3n-2b"  # Smaller model
        config.context_window = min(config.context_window, 16000)
        config.batch_size = 1
        config.precision = "int8"
        config.max_tokens = min(config.max_tokens, 256)
        config.cpu_offload = True
        
        return config
    
    def _estimate_memory_usage(self, config: GemmaConfig) -> float:
        """Estimate memory usage for configuration"""
        
        # Base model memory requirements (approximate)
        model_memory = {
            "gemma-3n-2b": 2.0,
            "gemma-3n-4b": 4.0,
            "gemma-3n-4b-hq": 6.0
        }
        
        base_memory = model_memory.get(config.model_variant, 4.0)
        
        # Precision multipliers
        precision_multiplier = {
            "fp32": 1.0,
            "fp16": 0.5,
            "int8": 0.25,
            "int4": 0.125
        }
        
        multiplier = precision_multiplier.get(config.precision, 0.5)
        
        # Context and batch adjustments
        context_factor = config.context_window / 32000  # Normalized to 32K
        batch_factor = config.batch_size
        
        estimated_memory = base_memory * multiplier * context_factor * batch_factor
        
        # Add overhead (20%)
        estimated_memory *= 1.2
        
        return estimated_memory
    
    def monitor_performance(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU metrics
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            
            if self.device_caps.gpu_available:
                try:
                    gpu_usage = self._get_gpu_usage()
                    gpu_memory_usage = self._get_gpu_memory_usage()
                except:
                    pass
            
            # Battery metrics
            battery = psutil.sensors_battery()
            battery_level = battery.percent if battery else 100.0
            
            # Temperature metrics
            temperature = self._get_cpu_temperature()
            
            # Performance metrics
            inference_speed = self._measure_inference_speed()
            power_consumption = self._estimate_power_consumption()
            network_latency = self._measure_network_latency()
            disk_io = self._get_disk_io()
            
            metrics = PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                battery_level=battery_level,
                temperature=temperature,
                inference_speed=inference_speed,
                power_consumption=power_consumption,
                network_latency=network_latency,
                disk_io=disk_io,
                timestamp=datetime.utcnow()
            )
            
            # Store in history
            self.performance_history.append(metrics)
            
            # Keep only last 100 measurements
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
            return PerformanceMetrics(
                cpu_usage=0, memory_usage=0, gpu_usage=0, gpu_memory_usage=0,
                battery_level=100, temperature=0, inference_speed=0,
                power_consumption=0, network_latency=0, disk_io=0,
                timestamp=datetime.utcnow()
            )
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization() or 0.0
            return 0.0
        except:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage"""
        try:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()
                memory_cached = torch.cuda.memory_reserved()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                return ((memory_allocated + memory_cached) / total_memory) * 100
            return 0.0
        except:
            return 0.0
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Try to find CPU temperature
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        return entries[0].current if entries else 0.0
                
                # Fallback to first available temperature
                first_sensor = list(temps.values())[0]
                return first_sensor[0].current if first_sensor else 0.0
            
            return 0.0
        except:
            return 0.0
    
    def _measure_inference_speed(self) -> float:
        """Measure inference speed (tokens per second)"""
        try:
            # This would measure actual inference speed
            # For now, return estimated speed based on config
            if self.current_config:
                if self.current_config.model_variant == "gemma-3n-2b":
                    base_speed = 15.0
                elif self.current_config.model_variant == "gemma-3n-4b":
                    base_speed = 10.0
                else:
                    base_speed = 7.0
                
                # Adjust for precision
                if self.current_config.precision == "int8":
                    base_speed *= 1.5
                elif self.current_config.precision == "int4":
                    base_speed *= 2.0
                
                # Adjust for GPU
                if self.device_caps.gpu_available:
                    base_speed *= 2.0
                
                return base_speed
            
            return 5.0  # Default estimate
        except:
            return 5.0
    
    def _estimate_power_consumption(self) -> float:
        """Estimate power consumption in watts"""
        try:
            # Simple estimation based on CPU and GPU usage
            cpu_power = (psutil.cpu_percent() / 100) * 50  # Assume max 50W for CPU
            gpu_power = 0.0
            
            if self.device_caps.gpu_available:
                gpu_power = (self._get_gpu_usage() / 100) * 150  # Assume max 150W for GPU
            
            return cpu_power + gpu_power
        except:
            return 50.0  # Default estimate
    
    def _measure_network_latency(self) -> float:
        """Measure network latency"""
        try:
            import subprocess
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'time=' in line:
                        time_str = line.split('time=')[1].split(' ')[0]
                        return float(time_str)
            
            return 100.0  # Default if ping fails
        except:
            return 100.0
    
    def _get_disk_io(self) -> float:
        """Get disk I/O usage"""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # Simple metric: read + write bytes per second
                return (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)  # MB/s
            return 0.0
        except:
            return 0.0
    
    def auto_adjust_settings(self, performance_metrics: PerformanceMetrics) -> Optional[GemmaConfig]:
        """Automatically adjust settings based on performance"""
        
        if not self.current_config:
            return None
        
        adjustments_made = False
        new_config = GemmaConfig(**asdict(self.current_config))
        
        # CPU usage too high
        if performance_metrics.cpu_usage > self.thresholds["cpu_critical"]:
            logger.warning(f"Critical CPU usage: {performance_metrics.cpu_usage}%")
            new_config = self._reduce_cpu_load(new_config)
            adjustments_made = True
        
        # Memory usage too high
        if performance_metrics.memory_usage > self.thresholds["memory_critical"]:
            logger.warning(f"Critical memory usage: {performance_metrics.memory_usage}%")
            new_config = self._reduce_memory_usage(new_config)
            adjustments_made = True
        
        # Battery level low
        if performance_metrics.battery_level < self.thresholds["battery_low"]:
            logger.warning(f"Low battery: {performance_metrics.battery_level}%")
            new_config = self._apply_power_saving(new_config)
            adjustments_made = True
        
        # Temperature too high
        if performance_metrics.temperature > self.thresholds["temperature_high"]:
            logger.warning(f"High temperature: {performance_metrics.temperature}°C")
            new_config = self._reduce_thermal_load(new_config)
            adjustments_made = True
        
        # Inference speed too slow
        if performance_metrics.inference_speed < self.thresholds["inference_slow"]:
            logger.warning(f"Slow inference: {performance_metrics.inference_speed} tokens/s")
            new_config = self._optimize_for_speed(new_config)
            adjustments_made = True
        
        if adjustments_made:
            self.current_config = new_config
            logger.info("Auto-adjusted AI settings for optimal performance")
            return new_config
        
        return None
    
    def _reduce_cpu_load(self, config: GemmaConfig) -> GemmaConfig:
        """Reduce CPU load"""
        config.batch_size = max(1, config.batch_size // 2)
        config.context_window = int(config.context_window * 0.8)
        config.cpu_offload = True
        return config
    
    def _reduce_memory_usage(self, config: GemmaConfig) -> GemmaConfig:
        """Reduce memory usage"""
        config.context_window = int(config.context_window * 0.7)
        config.batch_size = max(1, config.batch_size // 2)
        config.gradient_checkpointing = True
        if config.precision == "fp16":
            config.precision = "int8"
        return config
    
    def _reduce_thermal_load(self, config: GemmaConfig) -> GemmaConfig:
        """Reduce thermal load"""
        config.batch_size = max(1, config.batch_size // 2)
        config.max_tokens = min(config.max_tokens, 256)
        if config.model_variant == "gemma-3n-4b-hq":
            config.model_variant = "gemma-3n-4b"
        elif config.model_variant == "gemma-3n-4b":
            config.model_variant = "gemma-3n-2b"
        return config
    
    def _optimize_for_speed(self, config: GemmaConfig) -> GemmaConfig:
        """Optimize for inference speed"""
        config.precision = "int8"
        config.use_flash_attention = True
        config.gradient_checkpointing = False
        config.context_window = min(config.context_window, 32000)
        return config
    
    def start_continuous_monitoring(self, interval: int = 30):
        """Start continuous performance monitoring"""
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started continuous monitoring (interval: {interval}s)")
    
    def stop_continuous_monitoring(self):
        """Stop continuous performance monitoring"""
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped continuous monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Continuous monitoring loop"""
        
        while self.monitoring_active:
            try:
                metrics = self.monitor_performance()
                
                # Auto-adjust if needed
                adjusted_config = self.auto_adjust_settings(metrics)
                
                if adjusted_config:
                    logger.info("Performance-based auto-adjustment applied")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No performance data available"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_battery = sum(m.battery_level for m in recent_metrics) / len(recent_metrics)
        avg_temp = sum(m.temperature for m in recent_metrics) / len(recent_metrics)
        avg_inference_speed = sum(m.inference_speed for m in recent_metrics) / len(recent_metrics)
        
        # Find peaks
        max_cpu = max(m.cpu_usage for m in recent_metrics)
        max_memory = max(m.memory_usage for m in recent_metrics)
        max_temp = max(m.temperature for m in recent_metrics)
        min_battery = min(m.battery_level for m in recent_metrics)
        
        # Performance score (0-100)
        performance_score = self._calculate_performance_score(recent_metrics)
        
        return {
            "timeframe_hours": hours,
            "data_points": len(recent_metrics),
            "averages": {
                "cpu_usage": round(avg_cpu, 1),
                "memory_usage": round(avg_memory, 1),
                "gpu_usage": round(avg_gpu, 1),
                "battery_level": round(avg_battery, 1),
                "temperature": round(avg_temp, 1),
                "inference_speed": round(avg_inference_speed, 1)
            },
            "peaks": {
                "max_cpu_usage": round(max_cpu, 1),
                "max_memory_usage": round(max_memory, 1),
                "max_temperature": round(max_temp, 1),
                "min_battery_level": round(min_battery, 1)
            },
            "performance_score": performance_score,
            "recommendations": self._generate_performance_recommendations(recent_metrics),
            "current_config": asdict(self.current_config) if self.current_config else None
        }
    
    def _calculate_performance_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall performance score (0-100)"""
        
        if not metrics:
            return 0.0
        
        scores = []
        
        for metric in metrics:
            # CPU score (lower usage = higher score)
            cpu_score = max(0, 100 - metric.cpu_usage)
            
            # Memory score (lower usage = higher score)
            memory_score = max(0, 100 - metric.memory_usage)
            
            # Temperature score (lower temp = higher score)
            temp_score = max(0, 100 - (metric.temperature / 100 * 100))
            
            # Inference speed score (higher speed = higher score)
            speed_score = min(100, (metric.inference_speed / 20) * 100)
            
            # Battery score (higher level = higher score)
            battery_score = metric.battery_level
            
            # Weighted average
            overall_score = (
                cpu_score * 0.25 +
                memory_score * 0.25 +
                temp_score * 0.20 +
                speed_score * 0.20 +
                battery_score * 0.10
            )
            
            scores.append(overall_score)
        
        return round(sum(scores) / len(scores), 1)
    
    def _generate_performance_recommendations(self, metrics: List[PerformanceMetrics]) -> List[Dict[str, str]]:
        """Generate performance recommendations"""
        
        recommendations = []
        
        if not metrics:
            return recommendations
        
        avg_cpu = sum(m.cpu_usage for m in metrics) / len(metrics)
        avg_memory = sum(m.memory_usage for m in metrics) / len(metrics)
        avg_temp = sum(m.temperature for m in metrics) / len(metrics)
        avg_inference_speed = sum(m.inference_speed for m in metrics) / len(metrics)
        min_battery = min(m.battery_level for m in metrics)
        
        # CPU recommendations
        if avg_cpu > 80:
            recommendations.append({
                "type": "cpu",
                "priority": "high",
                "recommendation": "High CPU usage detected. Consider reducing model complexity or batch size.",
                "action": "reduce_cpu_load"
            })
        
        # Memory recommendations
        if avg_memory > 85:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "recommendation": "High memory usage detected. Reduce context window or enable gradient checkpointing.",
                "action": "reduce_memory_usage"
            })
        
        # Temperature recommendations
        if avg_temp > 75:
            recommendations.append({
                "type": "thermal",
                "priority": "medium",
                "recommendation": "Elevated temperature detected. Consider thermal throttling or reducing workload.",
                "action": "reduce_thermal_load"
            })
        
        # Speed recommendations
        if avg_inference_speed < 5:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "recommendation": "Slow inference speed detected. Consider using a smaller model or int8 precision.",
                "action": "optimize_for_speed"
            })
        
        # Battery recommendations
        if min_battery < 30:
            recommendations.append({
                "type": "power",
                "priority": "medium",
                "recommendation": "Low battery detected. Enable power saving mode.",
                "action": "enable_power_saving"
            })
        
        # No issues
        if not recommendations:
            recommendations.append({
                "type": "status",
                "priority": "info",
                "recommendation": "Performance is optimal. No adjustments needed.",
                "action": "maintain_current_settings"
            })
        
        return recommendations
    
    def export_performance_data(self, filename: Optional[str] = None) -> str:
        """Export performance data to JSON file"""
        
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "device_capabilities": asdict(self.device_caps),
            "current_config": asdict(self.current_config) if self.current_config else None,
            "performance_history": [
                {
                    **asdict(metric),
                    "timestamp": metric.timestamp.isoformat()
                }
                for metric in self.performance_history
            ],
            "thresholds": self.thresholds,
            "optimization_rules": self.optimization_rules
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            raise
    
    def load_performance_data(self, filename: str):
        """Load performance data from JSON file"""
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load performance history
            self.performance_history = []
            for metric_data in data.get("performance_history", []):
                metric_data["timestamp"] = datetime.fromisoformat(metric_data["timestamp"])
                self.performance_history.append(PerformanceMetrics(**metric_data))
            
            # Load configuration if available
            if data.get("current_config"):
                self.current_config = GemmaConfig(**data["current_config"])
            
            logger.info(f"Performance data loaded from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
            raise

# Global optimizer instance
adaptive_optimizer = AdaptiveAIOptimizer()

# Convenience functions for integration
def optimize_for_emergency() -> GemmaConfig:
    """Quick optimization for emergency use case"""
    return adaptive_optimizer.optimize_for_device("emergency")

def optimize_for_routine() -> GemmaConfig:
    """Quick optimization for routine use case"""
    return adaptive_optimizer.optimize_for_device("routine")

def get_current_performance() -> PerformanceMetrics:
    """Get current performance metrics"""
    return adaptive_optimizer.monitor_performance()

def start_monitoring(interval: int = 30):
    """Start continuous performance monitoring"""
    adaptive_optimizer.start_continuous_monitoring(interval)

def stop_monitoring():
    """Stop continuous performance monitoring"""
    adaptive_optimizer.stop_continuous_monitoring()

def get_device_info() -> DeviceCapabilities:
    """Get device capabilities information"""
    return adaptive_optimizer.device_caps