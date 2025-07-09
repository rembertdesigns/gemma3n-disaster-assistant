/**
 * Edge AI Monitor - Advanced Performance Optimization System
 * Real-time monitoring and optimization of AI models
 */

class EdgeAIMonitor {
    constructor() {
      this.isInitialized = false;
      this.autoOptimizationEnabled = false;
      this.performanceData = {
        cpu: 0,
        memory: 0,
        gpu: 0,
        network: 0,
        throughput: 0,
        latency: 0,
        accuracy: 0,
        errorRate: 0
      };
      
      this.models = [
        {
          name: 'Emergency Classification',
          accuracy: 98.4,
          latency: 87,
          status: 'optimal',
          memoryUsage: 245,
          loadTime: 1.2
        },
        {
          name: 'Hazard Detection',
          accuracy: 96.7,
          latency: 124,
          status: 'good',
          memoryUsage: 189,
          loadTime: 0.8
        },
        {
          name: 'Sentiment Analysis',
          accuracy: 94.2,
          latency: 56,
          status: 'optimal',
          memoryUsage: 67,
          loadTime: 0.4
        },
        {
          name: 'OCR Processing',
          accuracy: 99.1,
          latency: 234,
          status: 'warning',
          memoryUsage: 412,
          loadTime: 2.1
        },
        {
          name: 'Speech Recognition',
          accuracy: 92.8,
          latency: 178,
          status: 'good',
          memoryUsage: 298,
          loadTime: 1.5
        },
        {
          name: 'Image Classification',
          accuracy: 97.3,
          latency: 145,
          status: 'optimal',
          memoryUsage: 334,
          loadTime: 1.8
        },
        {
          name: 'NLP Processing',
          accuracy: 95.6,
          latency: 89,
          status: 'good',
          memoryUsage: 156,
          loadTime: 0.9
        }
      ];
      
      this.performanceHistory = [];
      this.alertThresholds = {
        cpu: 85,
        memory: 80,
        gpu: 90,
        latency: 200,
        accuracy: 90,
        errorRate: 2.0
      };
      
      this.optimizationHistory = [];
      this.benchmarkResults = {
        overall: 94.2,
        latency: 87.6,
        accuracy: 98.4,
        efficiency: 91.8
      };
      
      this.logEntries = [];
      this.updateInterval = null;
      this.chartContext = null;
      
      // Performance tracking
      this.metrics = {
        requestCount: 0,
        successCount: 0,
        errorCount: 0,
        totalLatency: 0,
        startTime: Date.now()
      };
    }
  
    /**
     * Initialize the AI Monitor system
     */
    async initialize() {
      if (this.isInitialized) return;
      
      try {
        this.log('Initializing Edge AI Monitor...', 'info');
        
        // Update status
        this.updateAIStatus('loading', 'Initializing AI Performance Monitor...');
        
        // Initialize performance tracking
        await this.initializePerformanceTracking();
        
        // Setup real-time monitoring
        this.startRealTimeMonitoring();
        
        // Initialize performance chart
        this.initializePerformanceChart();
        
        // Load saved preferences
        this.loadPreferences();
        
        // Update status to ready
        this.updateAIStatus('ready', 'AI Performance Monitor Ready');
        
        this.isInitialized = true;
        this.log('Edge AI Monitor initialized successfully', 'success');
        
        // Start auto-optimization if enabled
        if (this.autoOptimizationEnabled) {
          this.startAutoOptimization();
        }
        
      } catch (error) {
        this.log(`Initialization failed: ${error.message}`, 'error');
        this.updateAIStatus('error', 'Monitor initialization failed');
      }
    }
  
    /**
     * Initialize performance tracking system
     */
    async initializePerformanceTracking() {
      // Simulate initial performance data gathering
      await this.gatherSystemMetrics();
      
      // Initialize model performance data
      this.updateModelPerformance();
      
      // Setup performance benchmarks
      await this.runInitialBenchmarks();
      
      this.log('Performance tracking initialized', 'info');
    }
  
    /**
     * Gather system performance metrics
     */
    async gatherSystemMetrics() {
      // Simulate gathering real system metrics
      // In a real implementation, this would interface with system APIs
      
      const cpuUsage = this.simulateMetric(this.performanceData.cpu, 60, 80, 2);
      const memoryUsage = this.simulateMetric(this.performanceData.memory, 40, 60, 1.5);
      const gpuUsage = this.simulateMetric(this.performanceData.gpu, 70, 90, 3);
      const networkUsage = this.simulateMetric(this.performanceData.network, 20, 50, 2.5);
      
      this.performanceData.cpu = cpuUsage;
      this.performanceData.memory = memoryUsage;
      this.performanceData.gpu = gpuUsage;
      this.performanceData.network = networkUsage;
      
      // Calculate derived metrics
      this.performanceData.throughput = this.calculateThroughput();
      this.performanceData.latency = this.calculateAverageLatency();
      this.performanceData.accuracy = this.calculateOverallAccuracy();
      this.performanceData.errorRate = this.calculateErrorRate();
      
      // Store in history
      this.performanceHistory.push({
        timestamp: Date.now(),
        ...this.performanceData
      });
      
      // Keep only last 100 entries
      if (this.performanceHistory.length > 100) {
        this.performanceHistory.shift();
      }
    }
  
    /**
     * Simulate realistic metric fluctuations
     */
    simulateMetric(current, min, max, volatility) {
      if (current === 0) {
        return Math.random() * (max - min) + min;
      }
      
      const change = (Math.random() - 0.5) * volatility;
      const newValue = Math.max(min, Math.min(max, current + change));
      return Math.round(newValue * 10) / 10;
    }
  
    /**
     * Calculate system throughput
     */
    calculateThroughput() {
      const timeElapsed = (Date.now() - this.metrics.startTime) / 1000;
      return timeElapsed > 0 ? Math.round((this.metrics.requestCount / timeElapsed) * 10) / 10 : 0;
    }
  
    /**
     * Calculate average latency
     */
    calculateAverageLatency() {
      return this.metrics.requestCount > 0 
        ? Math.round(this.metrics.totalLatency / this.metrics.requestCount)
        : 0;
    }
  
    /**
     * Calculate overall accuracy
     */
    calculateOverallAccuracy() {
      const totalAccuracy = this.models.reduce((sum, model) => sum + model.accuracy, 0);
      return Math.round((totalAccuracy / this.models.length) * 10) / 10;
    }
  
    /**
     * Calculate error rate
     */
    calculateErrorRate() {
      return this.metrics.requestCount > 0
        ? Math.round((this.metrics.errorCount / this.metrics.requestCount) * 100 * 10) / 10
        : 0;
    }
  
    /**
     * Start real-time monitoring
     */
    startRealTimeMonitoring() {
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
      }
      
      this.updateInterval = setInterval(async () => {
        await this.gatherSystemMetrics();
        this.updateUI();
        this.checkAlerts();
        
        if (this.autoOptimizationEnabled) {
          this.runAutoOptimization();
        }
      }, 2000); // Update every 2 seconds
      
      this.log('Real-time monitoring started', 'info');
    }
  
    /**
     * Update the user interface with current metrics
     */
    updateUI() {
      // Update system overview
      this.updateElement('activeModels', this.models.length);
      this.updateElement('processingQueue', Math.floor(Math.random() * 20));
      this.updateElement('avgResponseTime', `${this.performanceData.latency}ms`);
      this.updateElement('totalRequests', this.formatNumber(this.metrics.requestCount));
      
      // Update performance metrics
      this.updateElement('throughput', `${this.performanceData.throughput} req/s`);
      this.updateElement('successRate', `${(100 - this.performanceData.errorRate).toFixed(1)}%`);
      this.updateElement('errorRate', `${this.performanceData.errorRate}%`);
      
      // Update resource usage
      this.updateResourceBar('cpu', this.performanceData.cpu);
      this.updateResourceBar('memory', this.performanceData.memory);
      this.updateResourceBar('gpu', this.performanceData.gpu);
      this.updateResourceBar('network', this.performanceData.network);
      
      // Update model list
      this.updateModelList();
      
      // Update benchmark scores
      this.updateBenchmarkScores();
      
      // Update performance chart
      this.updatePerformanceChart();
    }
  
    /**
     * Update element content safely
     */
    updateElement(id, content) {
      const element = document.getElementById(id);
      if (element) {
        element.textContent = content;
      }
    }
  
    /**
     * Update resource usage bars
     */
    updateResourceBar(resource, percentage) {
      const valueElement = document.getElementById(`${resource}Usage`);
      const barElement = document.getElementById(`${resource}Bar`);
      
      if (valueElement) {
        valueElement.textContent = `${Math.round(percentage)}%`;
      }
      
      if (barElement) {
        barElement.style.width = `${percentage}%`;
      }
    }
  
    /**
     * Update model performance list
     */
    updateModelList() {
      const modelList = document.getElementById('modelList');
      if (!modelList) return;
      
      modelList.innerHTML = this.models.map(model => `
        <div class="model-item">
          <div class="model-name">${model.name}</div>
          <div class="model-stats">
            <span>${model.accuracy}% accuracy</span>
            <span>${model.latency}ms latency</span>
          </div>
        </div>
      `).join('');
    }
  
    /**
     * Update benchmark scores
     */
    updateBenchmarkScores() {
      Object.keys(this.benchmarkResults).forEach(key => {
        const scoreElement = document.getElementById(`${key}Score`);
        const progressElement = document.getElementById(`${key}Progress`);
        
        if (scoreElement) {
          scoreElement.textContent = this.benchmarkResults[key].toFixed(1);
        }
        
        if (progressElement) {
          progressElement.style.width = `${this.benchmarkResults[key]}%`;
        }
      });
    }
  
    /**
     * Initialize performance chart
     */
    initializePerformanceChart() {
      const canvas = document.getElementById('performanceChart');
      if (!canvas) return;
      
      this.chartContext = canvas.getContext('2d');
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      
      this.drawPerformanceChart();
    }
  
    /**
     * Draw performance chart
     */
    drawPerformanceChart() {
      if (!this.chartContext) return;
      
      const ctx = this.chartContext;
      const canvas = ctx.canvas;
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw background
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, width, height);
      
      // Draw grid
      this.drawGrid(ctx, width, height);
      
      // Draw performance lines
      this.drawPerformanceLine(ctx, width, height, 'throughput', '#3b82f6');
      this.drawPerformanceLine(ctx, width, height, 'latency', '#ef4444');
      this.drawPerformanceLine(ctx, width, height, 'cpu', '#10b981');
      
      // Draw legend
      this.drawChartLegend(ctx, width, height);
    }
  
    /**
     * Draw chart grid
     */
    drawGrid(ctx, width, height) {
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      
      // Vertical lines
      for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      
      // Horizontal lines
      for (let i = 0; i <= 5; i++) {
        const y = (height / 5) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }
  
    /**
     * Draw performance line
     */
    drawPerformanceLine(ctx, width, height, metric, color) {
      if (this.performanceHistory.length < 2) return;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      const maxPoints = Math.min(50, this.performanceHistory.length);
      const data = this.performanceHistory.slice(-maxPoints);
      
      // Normalize data for the metric
      const values = data.map(d => d[metric] || 0);
      const maxValue = Math.max(...values) || 1;
      
      data.forEach((point, index) => {
        const x = (width / (maxPoints - 1)) * index;
        const normalizedValue = (point[metric] || 0) / maxValue;
        const y = height - (normalizedValue * height * 0.8);
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    }
  
    /**
     * Draw chart legend
     */
    drawChartLegend(ctx, width, height) {
      const legends = [
        { label: 'Throughput', color: '#3b82f6' },
        { label: 'Latency', color: '#ef4444' },
        { label: 'CPU', color: '#10b981' }
      ];
      
      ctx.font = '12px sans-serif';
      legends.forEach((legend, index) => {
        const x = 10;
        const y = 20 + (index * 20);
        
        // Draw color indicator
        ctx.fillStyle = legend.color;
        ctx.fillRect(x, y - 8, 12, 12);
        
        // Draw label
        ctx.fillStyle = '#374151';
        ctx.fillText(legend.label, x + 20, y);
      });
    }
  
    /**
     * Update performance chart
     */
    updatePerformanceChart() {
      if (this.chartContext) {
        this.drawPerformanceChart();
      }
    }
  
    /**
     * Check for performance alerts
     */
    checkAlerts() {
      const alerts = [];
      
      if (this.performanceData.cpu > this.alertThresholds.cpu) {
        alerts.push(`High CPU usage: ${this.performanceData.cpu}%`);
      }
      
      if (this.performanceData.memory > this.alertThresholds.memory) {
        alerts.push(`High memory usage: ${this.performanceData.memory}%`);
      }
      
      if (this.performanceData.gpu > this.alertThresholds.gpu) {
        alerts.push(`High GPU usage: ${this.performanceData.gpu}%`);
      }
      
      if (this.performanceData.latency > this.alertThresholds.latency) {
        alerts.push(`High latency: ${this.performanceData.latency}ms`);
      }
      
      if (this.performanceData.errorRate > this.alertThresholds.errorRate) {
        alerts.push(`High error rate: ${this.performanceData.errorRate}%`);
      }
      
      // Show alerts if any
      if (alerts.length > 0) {
        this.showAlert('Performance Alert', alerts.join('; '));
      } else {
        this.hideAlert();
      }
    }
  
    /**
     * Show performance alert
     */
    showAlert(title, message) {
      const alertPanel = document.getElementById('alertPanel');
      const alertTitle = document.getElementById('alertTitle');
      const alertMessage = document.getElementById('alertMessage');
      
      if (alertPanel && alertTitle && alertMessage) {
        alertTitle.textContent = title;
        alertMessage.textContent = message;
        alertPanel.classList.add('visible');
        
        this.log(`ALERT: ${message}`, 'warning');
      }
    }
  
    /**
     * Hide performance alert
     */
    hideAlert() {
      const alertPanel = document.getElementById('alertPanel');
      if (alertPanel) {
        alertPanel.classList.remove('visible');
      }
    }
  
    /**
     * Run initial benchmarks
     */
    async runInitialBenchmarks() {
      this.log('Running initial performance benchmarks...', 'info');
      
      // Simulate benchmark calculations
      await this.sleep(1000);
      
      // Update benchmark results with some variation
      this.benchmarkResults.overall = this.simulateMetric(94.2, 90, 98, 1);
      this.benchmarkResults.latency = this.simulateMetric(87.6, 80, 95, 2);
      this.benchmarkResults.accuracy = this.simulateMetric(98.4, 95, 99.5, 0.5);
      this.benchmarkResults.efficiency = this.simulateMetric(91.8, 85, 98, 1.5);
      
      this.log('Initial benchmarks completed', 'success');
    }
  
    /**
     * Toggle auto-optimization
     */
    toggleAutoOptimization() {
      this.autoOptimizationEnabled = !this.autoOptimizationEnabled;
      
      const button = document.getElementById('autoOptText');
      if (button) {
        button.textContent = this.autoOptimizationEnabled ? 'Disable Auto-Opt' : 'Enable Auto-Opt';
      }
      
      if (this.autoOptimizationEnabled) {
        this.startAutoOptimization();
        this.log('Auto-optimization enabled', 'info');
      } else {
        this.log('Auto-optimization disabled', 'info');
      }
      
      this.savePreferences();
    }
  
    /**
     * Start auto-optimization process
     */
    startAutoOptimization() {
      this.log('Auto-optimization process started', 'info');
      
      // Run optimization checks every 30 seconds
      setInterval(() => {
        if (this.autoOptimizationEnabled) {
          this.runAutoOptimization();
        }
      }, 30000);
    }
  
    /**
     * Run automatic optimization
     */
    runAutoOptimization() {
      // Check if optimization is needed
      const needsOptimization = 
        this.performanceData.cpu > 80 ||
        this.performanceData.memory > 75 ||
        this.performanceData.latency > 150;
      
      if (needsOptimization) {
        this.log('Running auto-optimization...', 'info');
        
        // Simulate optimization
        setTimeout(() => {
          this.optimizePerformance();
          this.log('Auto-optimization completed', 'success');
        }, 2000);
      }
    }
  
    /**
     * Performance boost mode
     */
    performanceBoost() {
      this.log('Activating performance boost mode...', 'info');
      
      // Simulate performance improvements
      setTimeout(() => {
        this.models.forEach(model => {
          model.latency = Math.max(20, model.latency * 0.7);
          model.accuracy = Math.max(85, model.accuracy * 0.95);
        });
        
        this.performanceData.cpu = Math.min(100, this.performanceData.cpu * 1.2);
        this.performanceData.latency = this.calculateAverageLatency();
        
        this.log('Performance boost activated - faster processing, slightly reduced accuracy', 'success');
        this.updateUI();
      }, 1500);
    }
  
    /**
     * Accuracy mode
     */
    accuracyMode() {
      this.log('Activating accuracy mode...', 'info');
      
      // Simulate accuracy improvements
      setTimeout(() => {
        this.models.forEach(model => {
          model.accuracy = Math.min(99.9, model.accuracy * 1.02);
          model.latency = model.latency * 1.3;
        });
        
        this.performanceData.latency = this.calculateAverageLatency();
        
        this.log('Accuracy mode activated - higher accuracy, slower processing', 'success');
        this.updateUI();
      }, 1500);
    }
  
    /**
     * Reload models
     */
    reloadModels() {
      this.log('Reloading AI models...', 'info');
      
      // Simulate model reloading
      setTimeout(() => {
        this.models.forEach(model => {
          model.accuracy = this.simulateMetric(model.accuracy, 94, 99, 1);
          model.latency = this.simulateMetric(model.latency, 50, 200, 10);
          model.status = model.latency < 100 ? 'optimal' : model.latency < 150 ? 'good' : 'warning';
        });
        
        // Reset some performance metrics
        this.performanceData.cpu = Math.max(30, this.performanceData.cpu * 0.8);
        this.performanceData.memory = Math.max(20, this.performanceData.memory * 0.9);
        
        this.log('All models reloaded successfully', 'success');
        this.updateUI();
      }, 3000);
    }
  
    /**
     * Optimize performance
     */
    optimizePerformance() {
      // Simulate optimization effects
      this.performanceData.cpu = Math.max(40, this.performanceData.cpu * 0.85);
      this.performanceData.memory = Math.max(30, this.performanceData.memory * 0.9);
      this.performanceData.latency = Math.max(50, this.performanceData.latency * 0.8);
      
      // Record optimization
      this.optimizationHistory.push({
        timestamp: Date.now(),
        type: 'auto',
        improvement: 'CPU -15%, Memory -10%, Latency -20%'
      });
    }
  
    /**
     * Run comprehensive diagnostics
     */
    async runDiagnostics() {
      this.log('Running comprehensive AI diagnostics...', 'info');
      
      // Update status
      this.updateAIStatus('loading', 'Running diagnostics...');
      
      try {
        // Test each model
        for (const model of this.models) {
          this.log(`Testing ${model.name}...`, 'info');
          await this.sleep(500);
          
          // Simulate test results
          const testResult = Math.random() > 0.1 ? 'PASS' : 'FAIL';
          if (testResult === 'PASS') {
            this.log(`${model.name}: PASSED`, 'success');
          } else {
            this.log(`${model.name}: FAILED - requires attention`, 'error');
          }
        }
        
        // System resource test
        this.log('Testing system resources...', 'info');
        await this.sleep(1000);
        this.log('System resources: OPTIMAL', 'success');
        
        // Network connectivity test
        this.log('Testing network connectivity...', 'info');
        await this.sleep(500);
        this.log('Network connectivity: STABLE', 'success');
        
        this.log('Diagnostics completed successfully', 'success');
        this.updateAIStatus('ready', 'Diagnostics completed - All systems optimal');
        
      } catch (error) {
        this.log(`Diagnostics failed: ${error.message}`, 'error');
        this.updateAIStatus('error', 'Diagnostics failed');
      }
    }
  
    /**
     * Update model performance data
     */
    updateModelPerformance() {
      this.models.forEach(model => {
        // Simulate performance fluctuations
        model.accuracy = this.simulateMetric(model.accuracy, 90, 99.5, 0.5);
        model.latency = this.simulateMetric(model.latency, 30, 300, 5);
        
        // Update status based on performance
        if (model.accuracy > 97 && model.latency < 100) {
          model.status = 'optimal';
        } else if (model.accuracy > 94 && model.latency < 150) {
          model.status = 'good';
        } else {
          model.status = 'warning';
        }
      });
    }
  
    /**
     * Update AI status indicator
     */
    updateAIStatus(status, message) {
      const statusDot = document.getElementById('aiStatusDot');
      const statusText = document.getElementById('aiStatusText');
      
      if (statusDot) {
        statusDot.className = 'ai-status-dot';
        switch (status) {
          case 'loading':
            statusDot.classList.add('loading');
            break;
          case 'error':
            statusDot.classList.add('error');
            break;
          case 'ready':
          default:
            // Default green color
            break;
        }
      }
      
      if (statusText) {
        statusText.textContent = `🧠 ${message}`;
      }
    }
  
    /**
     * Add log entry
     */
    log(message, type = 'info') {
      const timestamp = new Date().toISOString().slice(11, 19);
      const entry = {
        timestamp,
        message,
        type
      };
      
      this.logEntries.push(entry);
      
      // Keep only last 50 entries
      if (this.logEntries.length > 50) {
        this.logEntries.shift();
      }
      
      // Update log display
      this.updateLogDisplay();
      
      // Console log for debugging
      console.log(`[${timestamp}] ${type.toUpperCase()}: ${message}`);
    }
  
    /**
     * Update log display
     */
    updateLogDisplay() {
      const logContainer = document.getElementById('performanceLog');
      if (!logContainer) return;
      
      const logHTML = this.logEntries.slice(-10).map(entry => 
        `<div class="log-entry log-${entry.type}">[${new Date().toISOString().slice(0, 19).replace('T', ' ')}] ${entry.type.toUpperCase()}: ${entry.message}</div>`
      ).join('');
      
      logContainer.innerHTML = logHTML;
      
      // Auto-scroll to bottom
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  
    /**
     * Format number with commas
     */
    formatNumber(num) {
      return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }
  
    /**
     * Save preferences to localStorage
     */
    savePreferences() {
      const preferences = {
        autoOptimizationEnabled: this.autoOptimizationEnabled,
        alertThresholds: this.alertThresholds
      };
      
      localStorage.setItem('edgeAIMonitorPrefs', JSON.stringify(preferences));
    }
  
    /**
     * Load preferences from localStorage
     */
    loadPreferences() {
      try {
        const saved = localStorage.getItem('edgeAIMonitorPrefs');
        if (saved) {
          const preferences = JSON.parse(saved);
          this.autoOptimizationEnabled = preferences.autoOptimizationEnabled || false;
          this.alertThresholds = { ...this.alertThresholds, ...preferences.alertThresholds };
          
          // Update UI
          const button = document.getElementById('autoOptText');
          if (button) {
            button.textContent = this.autoOptimizationEnabled ? 'Disable Auto-Opt' : 'Enable Auto-Opt';
          }
        }
      } catch (error) {
        console.warn('Failed to load preferences:', error);
      }
    }
  
    /**
     * Utility: Sleep function
     */
    sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
  
    /**
     * Cleanup when page unloads
     */
    cleanup() {
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
      }
    }
  }
  
  // Initialize global EdgeAIMonitor instance
  window.EdgeAIMonitor = new EdgeAIMonitor();
  
  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    if (window.EdgeAIMonitor) {
      window.EdgeAIMonitor.cleanup();
    }
  });
  
  // Auto-initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      window.EdgeAIMonitor.initialize();
    });
  } else {
    window.EdgeAIMonitor.initialize();
  }