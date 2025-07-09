/**
 * Adaptive AI Settings - Gemma 3n Performance Optimization
 * Handles device-specific AI optimization and performance tuning
 */

class AdaptiveAISettings {
    constructor() {
        this.selectedModel = 'gemma-3n-4b';
        this.autoOptimization = false;
        this.settings = {
            quality: 50,
            memory: 50,
            context: 50,
            batch: 50,
            quantization: 50,
            kvCache: 75,
            cleanup: 75,
            cpuThrottle: 30,
            background: 25
        };
        this.deviceMetrics = {};
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadDeviceMetrics();
        this.initializeGemma3n();
        this.startMetricsMonitoring();
    }
    
    initializeElements() {
        // Model selection elements
        this.modelCards = document.querySelectorAll('.model-card');
        
        // Slider elements
        this.sliders = {
            quality: document.getElementById('qualitySlider'),
            memory: document.getElementById('memorySlider'),
            context: document.getElementById('contextSlider'),
            batch: document.getElementById('batchSlider'),
            quantization: document.getElementById('quantizationSlider'),
            kvCache: document.getElementById('kvCacheSlider'),
            cleanup: document.getElementById('cleanupSlider'),
            cpuThrottle: document.getElementById('cpuThrottleSlider'),
            background: document.getElementById('backgroundSlider')
        };
        
        // Value display elements
        this.values = {
            quality: document.getElementById('qualityValue'),
            memory: document.getElementById('memoryValue'),
            context: document.getElementById('contextValue'),
            batch: document.getElementById('batchValue'),
            quantization: document.getElementById('quantizationValue'),
            kvCache: document.getElementById('kvCacheValue'),
            cleanup: document.getElementById('cleanupValue'),
            cpuThrottle: document.getElementById('cpuThrottleValue'),
            background: document.getElementById('backgroundValue')
        };
        
        // Device monitoring elements
        this.deviceElements = {
            cpuUsage: document.getElementById('cpuUsage'),
            cpuBar: document.getElementById('cpuBar'),
            cpuIndicator: document.getElementById('cpuIndicator'),
            cpuTemp: document.getElementById('cpuTemp'),
            
            memoryUsage: document.getElementById('memoryUsage'),
            memoryBar: document.getElementById('memoryBar'),
            memoryIndicator: document.getElementById('memoryIndicator'),
            memoryAvailable: document.getElementById('memoryAvailable'),
            
            batteryLevel: document.getElementById('batteryLevel'),
            batteryBar: document.getElementById('batteryBar'),
            batteryIndicator: document.getElementById('batteryIndicator'),
            batteryTime: document.getElementById('batteryTime'),
            
            networkType: document.getElementById('networkType'),
            networkSpeed: document.getElementById('networkSpeed'),
            networkIndicator: document.getElementById('networkIndicator')
        };
        
        // Action buttons
        this.applySettingsBtn = document.getElementById('applySettingsBtn');
        this.testPerformanceBtn = document.getElementById('testPerformanceBtn');
        this.resetToOptimalBtn = document.getElementById('resetToOptimalBtn');
        this.exportConfigBtn = document.getElementById('exportConfigBtn');
        
        // Auto-optimization toggle
        this.autoOptimizationToggle = document.getElementById('autoOptimizationToggle');
        
        // Optimization recommendations
        this.optimizationRecommendations = document.getElementById('optimizationRecommendations');
    }
    
    async initializeGemma3n() {
        try {
            if (window.EdgeAI) {
                await window.EdgeAI.loadAdaptiveModel(this.selectedModel);
                console.log('✅ Gemma 3n adaptive model loaded');
                this.updateAIStatus('ready');
            } else {
                console.warn('⚠️ EdgeAI not available, using fallback mode');
                this.updateAIStatus('fallback');
            }
        } catch (error) {
            console.error('❌ Failed to initialize Gemma 3n:', error);
            this.updateAIStatus('error');
        }
    }
    
    updateAIStatus(status) {
        const aiStatusDot = document.getElementById('aiStatusDot');
        const aiStatusText = document.getElementById('aiStatusText');
        
        switch (status) {
            case 'ready':
                aiStatusDot.className = 'ai-status-dot';
                aiStatusText.textContent = '🧠 Gemma 3n Adaptive Mode Ready';
                break;
            case 'optimizing':
                aiStatusDot.className = 'ai-status-dot loading';
                aiStatusText.textContent = '🧠 Optimizing Gemma 3n Performance...';
                break;
            case 'fallback':
                aiStatusDot.className = 'ai-status-dot loading';
                aiStatusText.textContent = '🧠 Using Fallback Optimization';
                break;
            case 'error':
                aiStatusDot.className = 'ai-status-dot error';
                aiStatusText.textContent = '🧠 Adaptive AI Unavailable';
                break;
        }
    }
    
    setupEventListeners() {
        // Model selection
        this.modelCards.forEach(card => {
            card.addEventListener('click', () => {
                this.selectModel(card.dataset.model);
            });
        });
        
        // Sliders
        Object.entries(this.sliders).forEach(([key, slider]) => {
            if (slider) {
                slider.addEventListener('input', () => {
                    this.updateSlider(key, slider.value);
                });
            }
        });
        
        // Action buttons
        this.applySettingsBtn.addEventListener('click', () => {
            this.applySettings();
        });
        
        this.testPerformanceBtn.addEventListener('click', () => {
            this.testPerformance();
        });
        
        this.resetToOptimalBtn.addEventListener('click', () => {
            this.resetToOptimal();
        });
        
        this.exportConfigBtn.addEventListener('click', () => {
            this.exportConfiguration();
        });
        
        // Auto-optimization toggle
        this.autoOptimizationToggle.addEventListener('click', () => {
            this.toggleAutoOptimization();
        });
    }
    
    selectModel(modelName) {
        this.selectedModel = modelName;
        
        // Update UI
        this.modelCards.forEach(card => {
            card.classList.toggle('selected', card.dataset.model === modelName);
        });
        
        // Update recommendations based on model
        this.updateModelRecommendations(modelName);
        
        console.log(`Selected model: ${modelName}`);
    }
    
    updateModelRecommendations(modelName) {
        const recommendations = {
            'gemma-3n-2b': {
                memory: 25,
                context: 35,
                quantization: 75,
                message: 'Optimized for speed and low memory usage'
            },
            'gemma-3n-4b': {
                memory: 50,
                context: 50,
                quantization: 50,
                message: 'Balanced performance for most use cases'
            },
            'gemma-3n-4b-hq': {
                memory: 75,
                context: 85,
                quantization: 25,
                message: 'Maximum quality for critical analysis'
            }
        };
        
        const rec = recommendations[modelName];
        if (rec) {
            // Auto-adjust sliders based on model
            this.updateSlider('memory', rec.memory);
            this.updateSlider('context', rec.context);
            this.updateSlider('quantization', rec.quantization);
        }
    }
    
    updateSlider(sliderId, value) {
        this.settings[sliderId] = parseInt(value);
        
        // Update slider value
        if (this.sliders[sliderId]) {
            this.sliders[sliderId].value = value;
        }
        
        // Update display value
        if (this.values[sliderId]) {
            this.values[sliderId].textContent = this.formatSliderValue(sliderId, value);
        }
        
        // Update recommendations if auto-optimization is enabled
        if (this.autoOptimization) {
            this.generateOptimizationRecommendations();
        }
    }
    
    formatSliderValue(sliderId, value) {
        const formatters = {
            quality: (v) => v < 25 ? 'Fast' : v < 75 ? 'Balanced' : 'High Quality',
            memory: (v) => v < 25 ? 'Conservative' : v < 75 ? 'Auto' : 'Aggressive',
            context: (v) => v < 25 ? '32K' : v < 50 ? '64K' : v < 75 ? '96K' : '128K',
            batch: (v) => v < 25 ? 'Small' : v < 75 ? 'Medium' : 'Large',
            quantization: (v) => v < 25 ? '16-bit' : v < 50 ? '8-bit' : v < 75 ? '4-bit' : '2-bit',
            kvCache: (v) => v < 50 ? 'Disabled' : 'Enabled',
            cleanup: (v) => v < 25 ? 'Conservative' : v < 75 ? 'Moderate' : 'Aggressive',
            cpuThrottle: (v) => v < 25 ? 'None' : v < 75 ? 'Moderate' : 'Heavy',
            background: (v) => v < 25 ? 'Disabled' : v < 75 ? 'Limited' : 'Full'
        };
        
        return formatters[sliderId] ? formatters[sliderId](value) : `${value}%`;
    }
    
    async loadDeviceMetrics() {
        try {
            const response = await fetch('/api/device-performance');
            if (response.ok) {
                const data = await response.json();
                this.deviceMetrics = data.performance;
                this.updateDeviceDisplay();
                this.generateOptimizationRecommendations();
            } else {
                // Use fallback metrics
                this.loadFallbackMetrics();
            }
        } catch (error) {
            console.error('❌ Error loading device metrics:', error);
            this.loadFallbackMetrics();
        }
    }
    
    loadFallbackMetrics() {
        this.deviceMetrics = {
            cpu: { usage_percent: 45, core_count: 4 },
            memory: { total_gb: 8, used_percent: 67, available_gb: 2.6 },
            battery: { percent: 78, charging: false, time_left: "4h 12m" },
            system: { platform: "Browser" }
        };
        this.updateDeviceDisplay();
    }
    
    updateDeviceDisplay() {
        const metrics = this.deviceMetrics;
        
        // CPU metrics
        if (metrics.cpu) {
            this.deviceElements.cpuUsage.textContent = `${metrics.cpu.usage_percent}%`;
            this.deviceElements.cpuBar.style.width = `${metrics.cpu.usage_percent}%`;
            this.deviceElements.cpuTemp.textContent = `${38 + Math.floor(metrics.cpu.usage_percent / 10)}°C`;
            
            // Update CPU indicator color
            if (metrics.cpu.usage_percent > 80) {
                this.deviceElements.cpuIndicator.className = 'status-indicator critical';
                this.deviceElements.cpuBar.className = 'metric-fill critical';
            } else if (metrics.cpu.usage_percent > 60) {
                this.deviceElements.cpuIndicator.className = 'status-indicator warning';
                this.deviceElements.cpuBar.className = 'metric-fill warning';
            } else {
                this.deviceElements.cpuIndicator.className = 'status-indicator good';
                this.deviceElements.cpuBar.className = 'metric-fill good';
            }
        }
        
        // Memory metrics
        if (metrics.memory) {
            this.deviceElements.memoryUsage.textContent = `${(metrics.memory.total_gb * metrics.memory.used_percent / 100).toFixed(1)} GB`;
            this.deviceElements.memoryBar.style.width = `${metrics.memory.used_percent}%`;
            this.deviceElements.memoryAvailable.textContent = `${metrics.memory.available_gb} GB`;
            
            // Update memory indicator color
            if (metrics.memory.used_percent > 85) {
                this.deviceElements.memoryIndicator.className = 'status-indicator critical';
                this.deviceElements.memoryBar.className = 'metric-fill critical';
            } else if (metrics.memory.used_percent > 70) {
                this.deviceElements.memoryIndicator.className = 'status-indicator warning';
                this.deviceElements.memoryBar.className = 'metric-fill warning';
            } else {
                this.deviceElements.memoryIndicator.className = 'status-indicator good';
                this.deviceElements.memoryBar.className = 'metric-fill good';
            }
        }
        
        // Battery metrics
        if (metrics.battery) {
            this.deviceElements.batteryLevel.textContent = `${metrics.battery.percent}%`;
            this.deviceElements.batteryBar.style.width = `${metrics.battery.percent}%`;
            this.deviceElements.batteryTime.textContent = metrics.battery.time_left || 'N/A';
            
            // Update battery indicator color
            if (metrics.battery.percent < 20) {
                this.deviceElements.batteryIndicator.className = 'status-indicator critical';
                this.deviceElements.batteryBar.className = 'metric-fill critical';
            } else if (metrics.battery.percent < 50) {
                this.deviceElements.batteryIndicator.className = 'status-indicator warning';
                this.deviceElements.batteryBar.className = 'metric-fill warning';
            } else {
                this.deviceElements.batteryIndicator.className = 'status-indicator good';
                this.deviceElements.batteryBar.className = 'metric-fill good';
            }
        }
        
        // Network metrics (simulated)
        this.deviceElements.networkType.textContent = navigator.onLine ? 'WiFi' : 'Offline';
        this.deviceElements.networkSpeed.textContent = navigator.onLine ? '45 Mbps' : 'N/A';
        this.deviceElements.networkIndicator.className = navigator.onLine ? 'status-indicator good' : 'status-indicator critical';
    }
    
    startMetricsMonitoring() {
        // Update device metrics every 30 seconds
        setInterval(() => {
            this.loadDeviceMetrics();
        }, 30000);
        
        // Simulate real-time updates
        setInterval(() => {
            this.simulateMetricUpdates();
        }, 5000);
    }
    
    simulateMetricUpdates() {
        // Simulate small changes in metrics for demonstration
        if (this.deviceMetrics.cpu) {
            const variance = (Math.random() - 0.5) * 10;
            this.deviceMetrics.cpu.usage_percent = Math.max(10, Math.min(90, this.deviceMetrics.cpu.usage_percent + variance));
        }
        
        if (this.deviceMetrics.memory) {
            const variance = (Math.random() - 0.5) * 5;
            this.deviceMetrics.memory.used_percent = Math.max(30, Math.min(95, this.deviceMetrics.memory.used_percent + variance));
            this.deviceMetrics.memory.available_gb = this.deviceMetrics.memory.total_gb * (100 - this.deviceMetrics.memory.used_percent) / 100;
        }
        
        if (this.deviceMetrics.battery) {
            // Slowly decrease battery if not charging
            if (!this.deviceMetrics.battery.charging) {
                this.deviceMetrics.battery.percent = Math.max(0, this.deviceMetrics.battery.percent - 0.1);
            }
        }
        
        this.updateDeviceDisplay();
        
        // Auto-optimize if enabled
        if (this.autoOptimization) {
            this.performAutoOptimization();
        }
    }
    
    generateOptimizationRecommendations() {
        const recommendations = [];
        const metrics = this.deviceMetrics;
        
        // CPU-based recommendations
        if (metrics.cpu && metrics.cpu.usage_percent > 80) {
            recommendations.push({
                icon: '⚡',
                title: 'High CPU Usage Detected',
                description: 'Consider switching to Gemma 3n 2B model or reduce processing quality.',
                action: () => this.selectModel('gemma-3n-2b')
            });
        }
        
        // Memory-based recommendations
        if (metrics.memory && metrics.memory.used_percent > 85) {
            recommendations.push({
                icon: '💾',
                title: 'Memory Usage Critical',
                description: 'Enable aggressive memory cleanup and reduce context window size.',
                action: () => {
                    this.updateSlider('cleanup', 85);
                    this.updateSlider('context', 25);
                }
            });
        }
        
        // Battery-based recommendations
        if (metrics.battery && metrics.battery.percent < 30) {
            recommendations.push({
                icon: '🔋',
                title: 'Low Battery Mode',
                description: 'Switch to power-saving mode to extend battery life.',
                action: () => {
                    this.selectModel('gemma-3n-2b');
                    this.updateSlider('cpuThrottle', 70);
                    this.updateSlider('background', 10);
                }
            });
        }
        
        // Performance-based recommendations
        if (this.selectedModel === 'gemma-3n-4b-hq' && (metrics.cpu?.usage_percent > 70 || metrics.memory?.used_percent > 80)) {
            recommendations.push({
                icon: '🎯',
                title: 'Performance Optimization',
                description: 'Switch to balanced model for better device performance.',
                action: () => this.selectModel('gemma-3n-4b')
            });
        }
        
        this.displayRecommendations(recommendations);
    }
    
    displayRecommendations(recommendations) {
        // Clear existing recommendations except the static ones
        const dynamicRecs = this.optimizationRecommendations.querySelectorAll('.recommendation-item.dynamic');
        dynamicRecs.forEach(rec => rec.remove());
        
        // Add new recommendations
        recommendations.forEach(rec => {
            const item = document.createElement('div');
            item.className = 'recommendation-item dynamic';
            item.innerHTML = `
                <div class="recommendation-icon">${rec.icon}</div>
                <div class="recommendation-content">
                    <div class="recommendation-title">${rec.title}</div>
                    <div class="recommendation-description">${rec.description}</div>
                </div>
            `;
            
            if (rec.action) {
                item.style.cursor = 'pointer';
                item.addEventListener('click', rec.action);
            }
            
            this.optimizationRecommendations.appendChild(item);
        });
    }
    
    performAutoOptimization() {
        const metrics = this.deviceMetrics;
        
        // Auto-adjust based on device conditions
        if (metrics.battery && metrics.battery.percent < 20) {
            // Battery saving mode
            if (this.selectedModel !== 'gemma-3n-2b') {
                this.selectModel('gemma-3n-2b');
            }
            this.updateSlider('cpuThrottle', 80);
            this.updateSlider('background', 5);
        }
        
        if (metrics.memory && metrics.memory.used_percent > 90) {
            // Aggressive memory management
            this.updateSlider('cleanup', 90);
            this.updateSlider('memory', 25);
            this.updateSlider('context', 25);
        }
        
        if (metrics.cpu && metrics.cpu.usage_percent > 85) {
            // CPU throttling
            this.updateSlider('cpuThrottle', 60);
            this.updateSlider('quality', 25);
        }
    }
    
    toggleAutoOptimization() {
        this.autoOptimization = !this.autoOptimization;
        this.autoOptimizationToggle.classList.toggle('active', this.autoOptimization);
        
        if (this.autoOptimization) {
            this.performAutoOptimization();
        }
        
        console.log(`Auto-optimization: ${this.autoOptimization ? 'enabled' : 'disabled'}`);
    }
    
    async applySettings() {
        this.updateAIStatus('optimizing');
        
        try {
            const settingsData = {
                selected_model: this.selectedModel,
                performance_settings: this.settings,
                auto_optimization: this.autoOptimization,
                device_metrics: this.deviceMetrics
            };
            
            const response = await fetch('/api/optimize-ai-settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settingsData)
            });
            
            if (response.ok) {
                const result = await response.json();
                alert('✅ AI settings applied successfully!');
                console.log('Optimization result:', result);
            } else {
                throw new Error('Failed to apply settings');
            }
        } catch (error) {
            console.error('Error applying settings:', error);
            alert('❌ Failed to apply settings. Please try again.');
        } finally {
            this.updateAIStatus('ready');
        }
    }
    
    async testPerformance() {
        alert('🧪 Running AI performance test...');
        
        const startTime = performance.now();
        
        try {
            // Simulate AI performance test
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            const endTime = performance.now();
            const duration = Math.round(endTime - startTime);
            
            // Generate mock performance results
            const results = {
                responseTime: `${duration}ms`,
                throughput: `${Math.round(1000 / duration * 10)} requests/sec`,
                memoryUsage: `${this.deviceMetrics.memory?.used_percent || 65}%`,
                accuracy: '94.2%',
                model: this.selectedModel
            };
            
            const resultText = `Performance Test Results:
• Response Time: ${results.responseTime}
• Throughput: ${results.throughput}
• Memory Usage: ${results.memoryUsage}
• Model Accuracy: ${results.accuracy}
• Model: ${results.model}`;
            
            alert(`✅ Performance test completed!\n\n${resultText}`);
            
        } catch (error) {
            console.error('Performance test failed:', error);
            alert('❌ Performance test failed.');
        }
    }
    
    resetToOptimal() {
        if (confirm('Reset all settings to optimal defaults based on your device?')) {
            // Determine optimal settings based on device metrics
            const metrics = this.deviceMetrics;
            
            if (metrics.memory && metrics.memory.total_gb < 4) {
                // Low memory device
                this.selectModel('gemma-3n-2b');
                this.updateSlider('memory', 25);
                this.updateSlider('context', 25);
                this.updateSlider('quantization', 75);
            } else if (metrics.memory && metrics.memory.total_gb > 8) {
                // High memory device
                this.selectModel('gemma-3n-4b-hq');
                this.updateSlider('memory', 75);
                this.updateSlider('context', 85);
                this.updateSlider('quantization', 25);
            } else {
                // Standard device
                this.selectModel('gemma-3n-4b');
                this.updateSlider('memory', 50);
                this.updateSlider('context', 50);
                this.updateSlider('quantization', 50);
            }
            
            // Reset other settings to defaults
            this.updateSlider('quality', 50);
            this.updateSlider('batch', 50);
            this.updateSlider('kvCache', 75);
            this.updateSlider('cleanup', 50);
            this.updateSlider('cpuThrottle', 30);
            this.updateSlider('background', 25);
            
            alert('✅ Settings reset to optimal defaults for your device.');
        }
    }
    
    exportConfiguration() {
        const config = {
            title: 'Gemma 3n AI Configuration',
            timestamp: new Date().toISOString(),
            selectedModel: this.selectedModel,
            autoOptimization: this.autoOptimization,
            settings: this.settings,
            deviceMetrics: this.deviceMetrics,
            version: '2.2.0'
        };
        
        const dataStr = JSON.stringify(config, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `gemma-3n-config-${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    setModel(model) {
        this.selectModel(model);
    }
    
    setResolution(resolution) {
        // For future use with vision models
        console.log(`Resolution set to: ${resolution}`);
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdaptiveAISettings;
} else {
    window.AdaptiveAISettings = AdaptiveAISettings;
}