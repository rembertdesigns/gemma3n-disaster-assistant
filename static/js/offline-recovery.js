class OfflineRecoveryManager {
    constructor() {
      this.isOnline = navigator.onLine;
      this.syncAttempts = new Map(); // reportId -> attempt count
      this.conflictResolver = new ConflictResolver();
      this.stressTestRunner = new StressTestRunner();
      this.recoveryStrategies = new RecoveryStrategies();
      
      // Recovery settings
      this.config = {
        maxSyncAttempts: 5,
        syncRetryDelay: 2000, // 2 seconds
        backoffMultiplier: 1.5,
        maxRetryDelay: 30000, // 30 seconds
        conflictThreshold: 3, // attempts before manual review
        stuckReportTimeout: 300000, // 5 minutes
        recoveryCheckInterval: 10000 // 10 seconds
      };
      
      this.metrics = {
        syncSuccesses: 0,
        syncFailures: 0,
        conflictsResolved: 0,
        recoveryAttempts: 0,
        totalOfflineTime: 0,
        lastRecoveryCheck: Date.now()
      };
      
      this.init();
    }
  
    async init() {
      console.log('📶 Initializing Offline Recovery Manager...');
      
      // Setup network monitoring
      this.setupNetworkMonitoring();
      
      // Setup periodic recovery checks
      this.startRecoveryLoop();
      
      // Setup UI components
      this.setupUI();
      
      // Initial sync queue check
      await this.checkAndRecoverQueue();
      
      console.log('✅ Offline Recovery Manager ready');
    }
  
    setupNetworkMonitoring() {
      window.addEventListener('online', () => {
        console.log('🟢 Network online - triggering recovery');
        this.isOnline = true;
        this.onNetworkRecovered();
      });
  
      window.addEventListener('offline', () => {
        console.log('🔴 Network offline - entering offline mode');
        this.isOnline = false;
        this.onNetworkLost();
      });
  
      // Advanced connection quality monitoring
      if ('connection' in navigator) {
        navigator.connection.addEventListener('change', () => {
          this.assessConnectionQuality();
        });
      }
    }
  
    async onNetworkRecovered() {
      this.updateNetworkStatus(true);
      
      // Reset sync attempts for stuck reports
      this.syncAttempts.clear();
      
      // Trigger immediate sync
      await this.attemptFullRecovery();
      
      // Show recovery success
      this.showNotification('🟢 Network recovered - syncing reports...', 'success');
    }
  
    onNetworkLost() {
      this.updateNetworkStatus(false);
      this.showNotification('🔴 Network lost - reports will be queued for sync', 'warning');
    }
  
    startRecoveryLoop() {
      setInterval(async () => {
        await this.performRecoveryCheck();
      }, this.config.recoveryCheckInterval);
    }
  
    async performRecoveryCheck() {
      this.metrics.lastRecoveryCheck = Date.now();
      
      if (!this.isOnline) return;
      
      // Check for stuck reports
      await this.identifyStuckReports();
      
      // Attempt to recover failed syncs
      await this.recoverFailedSyncs();
      
      // Clean up old data
      await this.cleanupOldData();
      
      // Update metrics
      this.updateRecoveryMetrics();
    }
  
    async identifyStuckReports() {
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        const now = Date.now();
        const stuckReports = [];
  
        for (const report of queue) {
          const timeSinceCreated = now - new Date(report.timestamp).getTime();
          const attempts = this.syncAttempts.get(report.id) || 0;
          
          if (timeSinceCreated > this.config.stuckReportTimeout || 
              attempts >= this.config.maxSyncAttempts) {
            stuckReports.push({
              ...report,
              attempts,
              timeSinceCreated,
              reason: attempts >= this.config.maxSyncAttempts ? 'max_attempts' : 'timeout'
            });
          }
        }
  
        if (stuckReports.length > 0) {
          console.warn('⚠️ Found stuck reports:', stuckReports);
          await this.handleStuckReports(stuckReports);
        }
      } catch (error) {
        console.error('❌ Error identifying stuck reports:', error);
      }
    }
  
    async handleStuckReports(stuckReports) {
      for (const report of stuckReports) {
        console.log(`🔧 Attempting recovery for stuck report: ${report.id}`);
        
        try {
          // Try different recovery strategies
          const recovered = await this.recoveryStrategies.attemptRecovery(report);
          
          if (recovered) {
            console.log(`✅ Recovered stuck report: ${report.id}`);
            this.metrics.recoveryAttempts++;
            
            // Remove from stuck list
            await this.removeReportFromQueue(report.id);
            this.syncAttempts.delete(report.id);
          } else {
            // Move to manual review queue
            await this.moveToManualReview(report);
          }
        } catch (error) {
          console.error(`❌ Failed to recover report ${report.id}:`, error);
          await this.moveToManualReview(report);
        }
      }
    }
  
    async moveToManualReview(report) {
      try {
        let manualQueue = await idbKeyval.get('manual_review_queue') || [];
        manualQueue.push({
          ...report,
          reviewReason: 'recovery_failed',
          reviewTimestamp: new Date().toISOString(),
          attempts: this.syncAttempts.get(report.id) || 0
        });
        
        await idbKeyval.set('manual_review_queue', manualQueue);
        await this.removeReportFromQueue(report.id);
        
        this.showNotification(`⚠️ Report ${report.id.substring(0, 8)} needs manual review`, 'warning');
        this.updateManualReviewUI();
        
        console.log(`📋 Moved report to manual review: ${report.id}`);
      } catch (error) {
        console.error('❌ Failed to move report to manual review:', error);
      }
    }
  
    async recoverFailedSyncs() {
      if (!this.isOnline) return;
  
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        
        for (const report of queue) {
          const attempts = this.syncAttempts.get(report.id) || 0;
          
          if (attempts < this.config.maxSyncAttempts) {
            await this.attemptSyncWithBackoff(report);
          }
        }
      } catch (error) {
        console.error('❌ Error in recovery sync:', error);
      }
    }
  
    async attemptSyncWithBackoff(report) {
      const attempts = this.syncAttempts.get(report.id) || 0;
      const delay = Math.min(
        this.config.syncRetryDelay * Math.pow(this.config.backoffMultiplier, attempts),
        this.config.maxRetryDelay
      );
  
      setTimeout(async () => {
        try {
          console.log(`🔄 Retry sync attempt ${attempts + 1} for report: ${report.id}`);
          
          this.syncAttempts.set(report.id, attempts + 1);
          
          const success = await this.syncSingleReport(report);
          
          if (success) {
            console.log(`✅ Sync successful for report: ${report.id}`);
            this.metrics.syncSuccesses++;
            this.syncAttempts.delete(report.id);
            await this.removeReportFromQueue(report.id);
            this.updateSyncStatus();
          } else {
            console.warn(`⚠️ Sync failed for report: ${report.id}`);
            this.metrics.syncFailures++;
          }
        } catch (error) {
          console.error(`❌ Sync error for report ${report.id}:`, error);
          this.metrics.syncFailures++;
        }
      }, delay);
    }
  
    async syncSingleReport(report) {
      try {
        // Check for conflicts before syncing
        const conflict = await this.conflictResolver.checkForConflicts(report);
        
        if (conflict) {
          console.log(`⚠️ Conflict detected for report: ${report.id}`);
          const resolved = await this.conflictResolver.resolveConflict(report, conflict);
          
          if (!resolved) {
            await this.moveToManualReview(report);
            return false;
          }
          
          this.metrics.conflictsResolved++;
        }
        
        // Attempt sync
        const formData = new FormData();
        formData.append('json', JSON.stringify(report));
        
        if (report.has_offline_image && typeof getImageBlob === 'function') {
          const { blob } = await getImageBlob(report.id);
          if (blob) {
            formData.append('file', blob, report.image_metadata?.filename || 'image.jpg');
          }
        }
  
        const response = await fetch('/generate-report', {
          method: 'POST',
          body: formData,
          signal: AbortSignal.timeout(30000) // 30 second timeout
        });
  
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
  
        // Clean up on success
        if (report.has_offline_image && typeof deleteImageBlob === 'function') {
          await deleteImageBlob(report.id);
        }
  
        return true;
      } catch (error) {
        console.error('Sync error:', error);
        return false;
      }
    }
  
    async attemptFullRecovery() {
      console.log('🚀 Starting full recovery process...');
      
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        
        if (queue.length === 0) {
          console.log('✅ No reports to recover');
          return;
        }
  
        console.log(`🔄 Recovering ${queue.length} reports...`);
        
        // Show recovery progress
        this.showRecoveryProgress(0, queue.length);
        
        let successCount = 0;
        let failureCount = 0;
        
        for (let i = 0; i < queue.length; i++) {
          const report = queue[i];
          
          try {
            const success = await this.syncSingleReport(report);
            
            if (success) {
              successCount++;
              await this.removeReportFromQueue(report.id);
            } else {
              failureCount++;
            }
            
            // Update progress
            this.showRecoveryProgress(i + 1, queue.length);
            
            // Small delay to prevent overwhelming the server
            await new Promise(resolve => setTimeout(resolve, 500));
          } catch (error) {
            console.error(`Recovery error for report ${report.id}:`, error);
            failureCount++;
          }
        }
        
        // Show final results
        this.showRecoveryResults(successCount, failureCount);
        
      } catch (error) {
        console.error('❌ Full recovery failed:', error);
        this.showNotification('❌ Recovery process failed', 'error');
      }
    }
  
    async removeReportFromQueue(reportId) {
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        const updatedQueue = queue.filter(report => report.id !== reportId);
        await idbKeyval.set('sync_queue', updatedQueue);
        this.updateSyncStatus();
      } catch (error) {
        console.error('Error removing report from queue:', error);
      }
    }
  
    // UI Management
    setupUI() {
      this.createOfflineStatusPanel();
      this.createRecoveryControls();
      this.createManualReviewPanel();
    }
  
    createOfflineStatusPanel() {
      if (document.getElementById('offlineStatusPanel')) return;
  
      const panel = document.createElement('div');
      panel.id = 'offlineStatusPanel';
      panel.className = 'offline-status-panel';
      panel.innerHTML = `
        <div class="status-header">
          <span class="status-icon" id="networkStatusIcon">🟢</span>
          <span class="status-text" id="networkStatusText">Online</span>
          <button class="status-toggle" onclick="offlineRecovery.toggleNetworkSimulation()">
            📶 Simulate Offline
          </button>
        </div>
        <div class="status-details">
          <div class="status-metric">
            <span class="metric-label">Queue:</span>
            <span class="metric-value" id="queueCount">0</span>
          </div>
          <div class="status-metric">
            <span class="metric-label">Failed:</span>
            <span class="metric-value" id="failedCount">0</span>
          </div>
          <div class="status-metric">
            <span class="metric-label">Manual Review:</span>
            <span class="metric-value" id="manualReviewCount">0</span>
          </div>
        </div>
      `;
  
      // Add styles
      const style = document.createElement('style');
      style.textContent = `
        .offline-status-panel {
          position: fixed;
          top: 60px;
          right: 10px;
          background: rgba(0, 0, 0, 0.9);
          color: white;
          padding: 1rem;
          border-radius: 8px;
          font-family: monospace;
          font-size: 0.8rem;
          z-index: 9999;
          min-width: 200px;
        }
        .status-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.5rem;
        }
        .status-toggle {
          margin-left: auto;
          padding: 0.25rem 0.5rem;
          border: none;
          border-radius: 4px;
          background: #333;
          color: white;
          cursor: pointer;
          font-size: 0.7rem;
        }
        .status-details {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.5rem;
        }
        .status-metric {
          display: flex;
          justify-content: space-between;
        }
        .metric-value {
          font-weight: bold;
        }
        .recovery-controls {
          margin-top: 0.5rem;
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
        }
        .recovery-btn {
          padding: 0.25rem 0.5rem;
          border: none;
          border-radius: 4px;
          background: #2563eb;
          color: white;
          cursor: pointer;
          font-size: 0.7rem;
        }
        .recovery-progress {
          margin-top: 0.5rem;
          padding: 0.5rem;
          background: #1f2937;
          border-radius: 4px;
          display: none;
        }
        .progress-bar {
          width: 100%;
          height: 4px;
          background: #374151;
          border-radius: 2px;
          overflow: hidden;
          margin: 0.25rem 0;
        }
        .progress-fill {
          height: 100%;
          background: #10b981;
          transition: width 0.3s ease;
        }
      `;
      document.head.appendChild(style);
      document.body.appendChild(panel);
    }
  
    createRecoveryControls() {
      const panel = document.getElementById('offlineStatusPanel');
      if (!panel) return;
  
      const controls = document.createElement('div');
      controls.className = 'recovery-controls';
      controls.innerHTML = `
        <button class="recovery-btn" onclick="offlineRecovery.attemptFullRecovery()">
          🔄 Recover All
        </button>
        <button class="recovery-btn" onclick="offlineRecovery.clearQueue()">
          🗑️ Clear Queue
        </button>
        <button class="recovery-btn" onclick="offlineRecovery.runStressTest()">
          🧪 Stress Test
        </button>
      `;
      panel.appendChild(controls);
  
      // Add progress indicator
      const progress = document.createElement('div');
      progress.id = 'recoveryProgress';
      progress.className = 'recovery-progress';
      progress.innerHTML = `
        <div class="progress-text">Recovery in progress...</div>
        <div class="progress-bar">
          <div class="progress-fill" id="progressFill"></div>
        </div>
        <div class="progress-details" id="progressDetails">0 / 0 reports</div>
      `;
      panel.appendChild(progress);
    }
  
    createManualReviewPanel() {
      const existingPanel = document.getElementById('manualReviewPanel');
      if (existingPanel) return;
  
      const panel = document.createElement('div');
      panel.id = 'manualReviewPanel';
      panel.style.cssText = `
        position: fixed;
        bottom: 10px;
        right: 10px;
        background: rgba(255, 193, 7, 0.95);
        color: black;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.8rem;
        z-index: 9998;
        max-width: 300px;
        display: none;
      `;
      panel.innerHTML = `
        <div><strong>⚠️ Manual Review Required</strong></div>
        <div id="manualReviewList"></div>
        <button onclick="offlineRecovery.showManualReviewDetails()" 
                style="margin-top: 0.5rem; padding: 0.25rem 0.5rem;">
          📋 View Details
        </button>
      `;
      document.body.appendChild(panel);
    }
  
    // Network simulation for testing
    toggleNetworkSimulation() {
      if (this.networkSimulationActive) {
        this.stopNetworkSimulation();
      } else {
        this.startNetworkSimulation();
      }
    }
  
    startNetworkSimulation() {
      this.networkSimulationActive = true;
      this.originalFetch = window.fetch;
      
      // Override fetch to simulate network failures
      window.fetch = (...args) => {
        return Promise.reject(new Error('Simulated network failure'));
      };
      
      this.isOnline = false;
      this.onNetworkLost();
      
      document.getElementById('networkStatusIcon').textContent = '🔴';
      document.getElementById('networkStatusText').textContent = 'Offline (Simulated)';
      
      const toggle = document.querySelector('.status-toggle');
      toggle.textContent = '🟢 Restore Network';
      
      console.log('🔴 Network simulation started');
    }
  
    stopNetworkSimulation() {
      this.networkSimulationActive = false;
      
      if (this.originalFetch) {
        window.fetch = this.originalFetch;
      }
      
      this.isOnline = navigator.onLine;
      this.onNetworkRecovered();
      
      document.getElementById('networkStatusIcon').textContent = '🟢';
      document.getElementById('networkStatusText').textContent = 'Online';
      
      const toggle = document.querySelector('.status-toggle');
      toggle.textContent = '📶 Simulate Offline';
      
      console.log('🟢 Network simulation stopped');
    }
  
    // UI Updates
    updateNetworkStatus(online) {
      const icon = document.getElementById('networkStatusIcon');
      const text = document.getElementById('networkStatusText');
      
      if (icon && text) {
        icon.textContent = online ? '🟢' : '🔴';
        text.textContent = online ? 'Online' : 'Offline';
      }
    }
  
    async updateSyncStatus() {
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        const manualQueue = await idbKeyval.get('manual_review_queue') || [];
        
        const queueCount = document.getElementById('queueCount');
        const failedCount = document.getElementById('failedCount');
        const manualReviewCount = document.getElementById('manualReviewCount');
        
        if (queueCount) queueCount.textContent = queue.length;
        if (failedCount) failedCount.textContent = this.metrics.syncFailures;
        if (manualReviewCount) manualReviewCount.textContent = manualQueue.length;
        
        // Show/hide manual review panel
        const panel = document.getElementById('manualReviewPanel');
        if (panel) {
          panel.style.display = manualQueue.length > 0 ? 'block' : 'none';
        }
      } catch (error) {
        console.error('Error updating sync status:', error);
      }
    }
  
    updateManualReviewUI() {
      // Update the manual review panel with current items
      this.updateSyncStatus();
    }
  
    showRecoveryProgress(current, total) {
      const progress = document.getElementById('recoveryProgress');
      const fill = document.getElementById('progressFill');
      const details = document.getElementById('progressDetails');
      
      if (progress && fill && details) {
        progress.style.display = 'block';
        
        const percentage = total > 0 ? (current / total) * 100 : 0;
        fill.style.width = `${percentage}%`;
        details.textContent = `${current} / ${total} reports`;
        
        if (current === total) {
          setTimeout(() => {
            progress.style.display = 'none';
          }, 2000);
        }
      }
    }
  
    showRecoveryResults(successCount, failureCount) {
      const message = `Recovery complete: ${successCount} succeeded, ${failureCount} failed`;
      const type = failureCount === 0 ? 'success' : 'warning';
      this.showNotification(message, type);
    }
  
    showNotification(message, type = 'info') {
      // Create notification element
      const notification = document.createElement('div');
      notification.className = `notification notification-${type}`;
      notification.textContent = message;
      notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        z-index: 10000;
        animation: slideDown 0.3s ease;
        background: ${type === 'success' ? '#10b981' : 
                     type === 'warning' ? '#f59e0b' : 
                     type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
      `;
      
      // Add animation styles if not already added
      if (!document.getElementById('notificationStyles')) {
        const style = document.createElement('style');
        style.id = 'notificationStyles';
        style.textContent = `
          @keyframes slideDown {
            from { transform: translateX(-50%) translateY(-100%); opacity: 0; }
            to { transform: translateX(-50%) translateY(0); opacity: 1; }
          }
        `;
        document.head.appendChild(style);
      }
      
      document.body.appendChild(notification);
      
      // Remove after 5 seconds
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 5000);
    }
  
    // Utility methods
    async clearQueue() {
      if (confirm('Are you sure you want to clear the sync queue? This will delete all pending reports.')) {
        await idbKeyval.set('sync_queue', []);
        await idbKeyval.set('manual_review_queue', []);
        this.syncAttempts.clear();
        this.updateSyncStatus();
        this.showNotification('✅ Queue cleared', 'success');
      }
    }
  
    async runStressTest() {
      console.log('🧪 Starting stress test...');
      this.showNotification('🧪 Running offline stress test...', 'info');
      
      try {
        const results = await this.stressTestRunner.runFullTest();
        console.log('📊 Stress test results:', results);
        
        // Show results
        alert(`Stress Test Results:
          
  Reports Generated: ${results.reportsGenerated}
  Sync Success Rate: ${results.syncSuccessRate}%
  Average Recovery Time: ${results.avgRecoveryTime}ms
  Memory Usage: ${results.memoryUsage}
  Overall Score: ${results.overallScore}/100`);
        
      } catch (error) {
        console.error('❌ Stress test failed:', error);
        this.showNotification('❌ Stress test failed', 'error');
      }
    }
  
    showManualReviewDetails() {
      // Open detailed manual review interface
      window.open('/manual-review', '_blank');
    }
  
    // Metrics and monitoring
    updateRecoveryMetrics() {
      // Update internal metrics for monitoring
      this.metrics.lastRecoveryCheck = Date.now();
    }
  
    getMetrics() {
      return {
        ...this.metrics,
        queueSize: this.syncAttempts.size,
        configuredMaxAttempts: this.config.maxSyncAttempts,
        isOnline: this.isOnline
      };
    }
  
    async checkAndRecoverQueue() {
      // Initial check on startup
      await this.performRecoveryCheck();
      await this.updateSyncStatus();
    }
  
    assessConnectionQuality() {
      if (!('connection' in navigator)) return;
      
      const connection = navigator.connection;
      const quality = {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
        saveData: connection.saveData
      };
      
      console.log('📶 Connection quality:', quality);
      
      // Adjust sync strategy based on connection quality
      if (quality.effectiveType === 'slow-2g' || quality.downlink < 0.5) {
        this.config.syncRetryDelay = 5000; // Longer delays for slow connections
        this.config.maxRetryDelay = 60000;
      } else {
        this.config.syncRetryDelay = 2000; // Default delays
        this.config.maxRetryDelay = 30000;
      }
    }
  
    async cleanupOldData() {
      try {
        const cutoffTime = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days ago
        
        // Clean up old manual review items
        const manualQueue = await idbKeyval.get('manual_review_queue') || [];
        const filteredQueue = manualQueue.filter(item => {
          return new Date(item.reviewTimestamp).getTime() > cutoffTime;
        });
        
        if (filteredQueue.length !== manualQueue.length) {
          await idbKeyval.set('manual_review_queue', filteredQueue);
          console.log(`🧹 Cleaned up ${manualQueue.length - filteredQueue.length} old manual review items`);
        }
      } catch (error) {
        console.error('Error cleaning up old data:', error);
      }
    }
  }
  
  // Conflict Resolution System
  class ConflictResolver {
    async checkForConflicts(report) {
      // Simulate conflict detection
      // In production, this would check server state
      return null; // No conflicts for now
    }
  
    async resolveConflict(report, conflict) {
      // Automatic conflict resolution strategies
      console.log('🔧 Resolving conflict for report:', report.id);
      
      // Strategy 1: Timestamp-based resolution
      if (conflict.type === 'timestamp') {
        return this.resolveByTimestamp(report, conflict);
      }
      
      // Strategy 2: Severity-based resolution
      if (conflict.type === 'severity') {
        return this.resolveBySeverity(report, conflict);
      }
      
      // Default: require manual resolution
      return false;
    }
  
    resolveByTimestamp(report, conflict) {
      // Use most recent version
      return new Date(report.timestamp) > new Date(conflict.serverTimestamp);
    }
  
    resolveBySeverity(report, conflict) {
      // Use higher severity version
      return report.severity > conflict.serverSeverity;
    }
  }
  
  // Recovery Strategies
  class RecoveryStrategies {
    async attemptRecovery(report) {
      console.log(`🔧 Attempting recovery for report: ${report.id}`);
      
      // Strategy 1: Retry with simplified payload
      try {
        const simplified = this.createSimplifiedReport(report);
        const success = await this.syncSimplifiedReport(simplified);
        if (success) return true;
      } catch (error) {
        console.warn('Simplified sync failed:', error);
      }
      
      // Strategy 2: Split large reports
      if (this.isLargeReport(report)) {
        try {
          const success = await this.syncInParts(report);
          if (success) return true;
        } catch (error) {
          console.warn('Part-based sync failed:', error);
        }
      }
      
      // Strategy 3: Compress and retry
      try {
        const compressed = await this.compressReport(report);
        const success = await this.syncCompressedReport(compressed);
        if (success) return true;
      } catch (error) {
        console.warn('Compressed sync failed:', error);
      }
      
      return false;
    }
  
    createSimplifiedReport(report) {
      // Remove non-essential data
      return {
        id: report.id,
        timestamp: report.timestamp,
        location: report.location,
        coordinates: report.coordinates,
        severity: report.severity,
        notes: report.notes.substring(0, 500), // Truncate notes
        hazards: report.hazards.slice(0, 3), // Limit hazards
        // Remove: image_url, ai_analysis, detailed metadata
        simplified: true
      };
    }
  
    isLargeReport(report) {
      const reportSize = JSON.stringify(report).length;
      return reportSize > 50000; // 50KB threshold
    }
  
    async syncInParts(report) {
      // Split report into metadata and attachments
      const metadata = { ...report };
      delete metadata.image_url;
      delete metadata.ai_analysis;
      
      // Sync metadata first
      const metadataSuccess = await this.syncReportPart(metadata, 'metadata');
      if (!metadataSuccess) return false;
      
      // Sync attachments separately if they exist
      if (report.has_offline_image) {
        const imageSuccess = await this.syncReportPart(
          { id: report.id, type: 'image' }, 'image'
        );
        if (!imageSuccess) return false;
      }
      
      return true;
    }
  
    async syncReportPart(partData, partType) {
      try {
        const response = await fetch(`/sync-part/${partType}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(partData),
          signal: AbortSignal.timeout(15000)
        });
        
        return response.ok;
      } catch (error) {
        console.error(`Failed to sync ${partType}:`, error);
        return false;
      }
    }
  
    async compressReport(report) {
      // Simple compression by removing whitespace and redundant data
      const compressed = {
        ...report,
        notes: report.notes.replace(/\s+/g, ' ').trim(),
        compressed: true
      };
      
      // Remove empty fields
      Object.keys(compressed).forEach(key => {
        if (compressed[key] === '' || compressed[key] === null || compressed[key] === undefined) {
          delete compressed[key];
        }
      });
      
      return compressed;
    }
  
    async syncSimplifiedReport(report) {
      try {
        const response = await fetch('/sync-simplified', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(report),
          signal: AbortSignal.timeout(10000)
        });
        
        return response.ok;
      } catch (error) {
        console.error('Simplified sync failed:', error);
        return false;
      }
    }
  
    async syncCompressedReport(report) {
      try {
        const response = await fetch('/sync-compressed', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Content-Encoding': 'simplified'
          },
          body: JSON.stringify(report),
          signal: AbortSignal.timeout(10000)
        });
        
        return response.ok;
      } catch (error) {
        console.error('Compressed sync failed:', error);
        return false;
      }
    }
  }
  
  // Stress Testing Suite
  class StressTestRunner {
    constructor() {
      this.testConfig = {
        reportCount: 50,
        imageSize: 1024 * 1024, // 1MB
        networkFailureRate: 0.3,
        testDuration: 30000, // 30 seconds
        concurrentRequests: 5
      };
    }
  
    async runFullTest() {
      console.log('🧪 Starting comprehensive stress test...');
      
      const startTime = Date.now();
      const initialMemory = this.getMemoryUsage();
      
      const results = {
        startTime,
        reportsGenerated: 0,
        syncAttempts: 0,
        syncSuccesses: 0,
        syncFailures: 0,
        conflicts: 0,
        recoveries: 0,
        avgResponseTime: 0,
        memoryUsage: initialMemory,
        errors: []
      };
  
      try {
        // Test 1: Generate flood of reports
        await this.testReportFlood(results);
        
        // Test 2: Network instability simulation
        await this.testNetworkInstability(results);
        
        // Test 3: Storage capacity test
        await this.testStorageCapacity(results);
        
        // Test 4: Recovery mechanisms
        await this.testRecoveryMechanisms(results);
        
        // Test 5: Memory stress test
        await this.testMemoryStress(results);
        
      } catch (error) {
        results.errors.push(error.message);
        console.error('Stress test error:', error);
      }
  
      // Calculate final metrics
      results.endTime = Date.now();
      results.totalDuration = results.endTime - results.startTime;
      results.syncSuccessRate = results.syncAttempts > 0 ? 
        (results.syncSuccesses / results.syncAttempts) * 100 : 0;
      results.avgRecoveryTime = results.totalDuration / Math.max(results.recoveries, 1);
      results.finalMemory = this.getMemoryUsage();
      results.memoryIncrease = this.calculateMemoryIncrease(initialMemory, results.finalMemory);
      results.overallScore = this.calculateOverallScore(results);
  
      console.log('📊 Stress test completed:', results);
      return results;
    }
  
    async testReportFlood(results) {
      console.log('🌊 Testing report flood...');
      
      const promises = [];
      for (let i = 0; i < this.testConfig.reportCount; i++) {
        promises.push(this.generateTestReport(i));
        results.reportsGenerated++;
        
        // Batch requests to avoid overwhelming
        if (promises.length >= this.testConfig.concurrentRequests) {
          await Promise.allSettled(promises);
          promises.length = 0;
          await new Promise(resolve => setTimeout(resolve, 100)); // Small delay
        }
      }
      
      // Handle remaining promises
      if (promises.length > 0) {
        await Promise.allSettled(promises);
      }
    }
  
    async testNetworkInstability(results) {
      console.log('📶 Testing network instability...');
      
      // Simulate intermittent network failures
      const originalFetch = window.fetch;
      window.fetch = (...args) => {
        if (Math.random() < this.testConfig.networkFailureRate) {
          results.syncFailures++;
          return Promise.reject(new Error('Simulated network failure'));
        }
        results.syncAttempts++;
        return originalFetch(...args).then(response => {
          if (response.ok) results.syncSuccesses++;
          else results.syncFailures++;
          return response;
        }).catch(error => {
          results.syncFailures++;
          throw error;
        });
      };
      
      // Generate reports during network instability
      for (let i = 0; i < 20; i++) {
        await this.generateTestReport(`unstable_${i}`);
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      // Restore normal fetch
      window.fetch = originalFetch;
    }
  
    async testStorageCapacity(results) {
      console.log('💾 Testing storage capacity...');
      
      try {
        // Generate large reports to test storage limits
        const largeReports = [];
        for (let i = 0; i < 10; i++) {
          const largeReport = await this.generateLargeTestReport(i);
          largeReports.push(largeReport);
          
          // Try to store in IndexedDB
          try {
            await idbKeyval.set(`stress_test_${i}`, largeReport);
          } catch (error) {
            results.errors.push(`Storage capacity exceeded at report ${i}: ${error.message}`);
            break;
          }
        }
        
        // Clean up test data
        for (let i = 0; i < 10; i++) {
          await idbKeyval.del(`stress_test_${i}`);
        }
        
      } catch (error) {
        results.errors.push(`Storage test failed: ${error.message}`);
      }
    }
  
    async testRecoveryMechanisms(results) {
      console.log('🔧 Testing recovery mechanisms...');
      
      // Create stuck reports
      const stuckReports = [];
      for (let i = 0; i < 5; i++) {
        const stuckReport = await this.generateTestReport(`stuck_${i}`);
        stuckReport.timestamp = new Date(Date.now() - 600000).toISOString(); // 10 minutes ago
        stuckReports.push(stuckReport);
      }
      
      // Add to queue
      const currentQueue = await idbKeyval.get('sync_queue') || [];
      await idbKeyval.set('sync_queue', [...currentQueue, ...stuckReports]);
      
      // Test recovery
      const recoveryStart = Date.now();
      if (window.offlineRecovery) {
        await window.offlineRecovery.identifyStuckReports();
        results.recoveries += stuckReports.length;
      }
      const recoveryTime = Date.now() - recoveryStart;
      
      console.log(`Recovery test completed in ${recoveryTime}ms`);
    }
  
    async testMemoryStress(results) {
      console.log('🧠 Testing memory stress...');
      
      const memoryHogs = [];
      try {
        // Create memory-intensive operations
        for (let i = 0; i < 100; i++) {
          // Create large objects
          const largeObject = new Array(10000).fill().map(() => ({
            id: crypto.randomUUID(),
            data: new Array(1000).fill(Math.random()),
            timestamp: new Date().toISOString()
          }));
          memoryHogs.push(largeObject);
          
          // Check memory periodically
          if (i % 20 === 0) {
            const currentMemory = this.getMemoryUsage();
            console.log(`Memory check ${i}: ${currentMemory}`);
          }
        }
        
        // Force garbage collection if available
        if (window.gc) {
          window.gc();
        }
        
      } catch (error) {
        results.errors.push(`Memory stress test failed: ${error.message}`);
      } finally {
        // Clean up
        memoryHogs.length = 0;
      }
    }
  
    async generateTestReport(id) {
      const testReport = {
        id: `stress_test_${id}_${crypto.randomUUID()}`,
        timestamp: new Date().toISOString(),
        location: `Test Location ${id}`,
        coordinates: [30.2672 + (Math.random() - 0.5) * 0.1, -97.7431 + (Math.random() - 0.5) * 0.1],
        hazards: ['fire', 'structural_damage', 'debris'].slice(0, Math.floor(Math.random() * 3) + 1),
        severity: Math.floor(Math.random() * 10) + 1,
        notes: `Stress test report ${id}. ${new Array(100).fill('Test data').join(' ')}`,
        stress_test: true,
        ai_analysis: {
          confidence: Math.random(),
          processing_time: Math.random() * 1000,
          fake_data: new Array(50).fill().map(() => Math.random())
        }
      };
  
      // Add to queue
      const queue = await idbKeyval.get('sync_queue') || [];
      queue.push(testReport);
      await idbKeyval.set('sync_queue', queue);
  
      return testReport;
    }
  
    async generateLargeTestReport(id) {
      const baseReport = await this.generateTestReport(id);
      
      // Add large data structures
      baseReport.large_data = {
        coordinates_history: new Array(1000).fill().map(() => [
          Math.random() * 180 - 90,
          Math.random() * 360 - 180
        ]),
        sensor_readings: new Array(5000).fill().map(() => ({
          timestamp: Date.now(),
          temperature: Math.random() * 100,
          humidity: Math.random() * 100,
          pressure: Math.random() * 1013 + 900
        })),
        image_metadata: {
          large_binary_data: new Array(10000).fill(0).map(() => Math.floor(Math.random() * 256))
        }
      };
  
      return baseReport;
    }
  
    getMemoryUsage() {
      if (performance.memory) {
        return {
          used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
          total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
          limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
        };
      }
      return { used: 0, total: 0, limit: 0 };
    }
  
    calculateMemoryIncrease(initial, final) {
      if (typeof initial === 'object' && typeof final === 'object') {
        return final.used - initial.used;
      }
      return 0;
    }
  
    calculateOverallScore(results) {
      let score = 100;
      
      // Deduct points for failures
      if (results.syncSuccessRate < 90) score -= (90 - results.syncSuccessRate);
      if (results.errors.length > 0) score -= results.errors.length * 5;
      if (results.memoryIncrease > 50) score -= 10; // Penalize memory leaks
      if (results.avgRecoveryTime > 5000) score -= 10; // Penalize slow recovery
      
      return Math.max(0, Math.round(score));
    }
  }
  
  // Initialize the offline recovery system
  window.offlineRecovery = new OfflineRecoveryManager();
  
  // Global utilities for integration
  window.OfflineStressTesting = {
    runStressTest: () => window.offlineRecovery.runStressTest(),
    simulateNetworkFailure: () => window.offlineRecovery.startNetworkSimulation(),
    restoreNetwork: () => window.offlineRecovery.stopNetworkSimulation(),
    clearAllQueues: () => window.offlineRecovery.clearQueue(),
    getMetrics: () => window.offlineRecovery.getMetrics(),
    forceRecovery: () => window.offlineRecovery.attemptFullRecovery()
  };
  
  console.log('📶 Offline Recovery & Stress Testing System loaded!');
  console.log('🧪 Use OfflineStressTesting.runStressTest() to start comprehensive testing');
  console.log('📊 Use OfflineStressTesting.getMetrics() to view current metrics');/**
   * 📶 Offline Stress Testing & Recovery System
   * 
   * File: /static/js/offline-recovery.js
   * 
   * Comprehensive offline resilience for emergency response scenarios
   * - Network failure simulation
   * - Sync conflict resolution  
   * - Visual feedback for stuck reports
   * - Automatic recovery mechanisms
   * - Stress testing suite
   */
  
  class OfflineRecoveryManager {
    constructor() {
      this.isOnline = navigator.onLine;
      this.syncAttempts = new Map(); // reportId -> attempt count
      this.conflictResolver = new ConflictResolver();
      this.stressTestRunner = new StressTestRunner();
      this.recoveryStrategies = new RecoveryStrategies();
      
      // Recovery settings
      this.config = {
        maxSyncAttempts: 5,
        syncRetryDelay: 2000, // 2 seconds
        backoffMultiplier: 1.5,
        maxRetryDelay: 30000, // 30 seconds
        conflictThreshold: 3, // attempts before manual review
        stuckReportTimeout: 300000, // 5 minutes
        recoveryCheckInterval: 10000 // 10 seconds
      };
      
      this.metrics = {
        syncSuccesses: 0,
        syncFailures: 0,
        conflictsResolved: 0,
        recoveryAttempts: 0,
        totalOfflineTime: 0,
        lastRecoveryCheck: Date.now()
      };
      
      this.init();
    }
  
    async init() {
      console.log('📶 Initializing Offline Recovery Manager...');
      
      // Setup network monitoring
      this.setupNetworkMonitoring();
      
      // Setup periodic recovery checks
      this.startRecoveryLoop();
      
      // Setup UI components
      this.setupUI();
      
      // Initial sync queue check
      await this.checkAndRecoverQueue();
      
      console.log('✅ Offline Recovery Manager ready');
    }
  
    setupNetworkMonitoring() {
      window.addEventListener('online', () => {
        console.log('🟢 Network online - triggering recovery');
        this.isOnline = true;
        this.onNetworkRecovered();
      });
  
      window.addEventListener('offline', () => {
        console.log('🔴 Network offline - entering offline mode');
        this.isOnline = false;
        this.onNetworkLost();
      });
  
      // Advanced connection quality monitoring
      if ('connection' in navigator) {
        navigator.connection.addEventListener('change', () => {
          this.assessConnectionQuality();
        });
      }
    }
  
    async onNetworkRecovered() {
      this.updateNetworkStatus(true);
      
      // Reset sync attempts for stuck reports
      this.syncAttempts.clear();
      
      // Trigger immediate sync
      await this.attemptFullRecovery();
      
      // Show recovery success
      this.showNotification('🟢 Network recovered - syncing reports...', 'success');
    }
  
    onNetworkLost() {
      this.updateNetworkStatus(false);
      this.showNotification('🔴 Network lost - reports will be queued for sync', 'warning');
    }
  
    startRecoveryLoop() {
      setInterval(async () => {
        await this.performRecoveryCheck();
      }, this.config.recoveryCheckInterval);
    }
  
    async performRecoveryCheck() {
      this.metrics.lastRecoveryCheck = Date.now();
      
      if (!this.isOnline) return;
      
      // Check for stuck reports
      await this.identifyStuckReports();
      
      // Attempt to recover failed syncs
      await this.recoverFailedSyncs();
      
      // Clean up old data
      await this.cleanupOldData();
      
      // Update metrics
      this.updateRecoveryMetrics();
    }
  
    async identifyStuckReports() {
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        const now = Date.now();
        const stuckReports = [];
  
        for (const report of queue) {
          const timeSinceCreated = now - new Date(report.timestamp).getTime();
          const attempts = this.syncAttempts.get(report.id) || 0;
          
          if (timeSinceCreated > this.config.stuckReportTimeout || 
              attempts >= this.config.maxSyncAttempts) {
            stuckReports.push({
              ...report,
              attempts,
              timeSinceCreated,
              reason: attempts >= this.config.maxSyncAttempts ? 'max_attempts' : 'timeout'
            });
          }
        }
  
        if (stuckReports.length > 0) {
          console.warn('⚠️ Found stuck reports:', stuckReports);
          await this.handleStuckReports(stuckReports);
        }
      } catch (error) {
        console.error('❌ Error identifying stuck reports:', error);
      }
    }
  
    async handleStuckReports(stuckReports) {
      for (const report of stuckReports) {
        console.log(`🔧 Attempting recovery for stuck report: ${report.id}`);
        
        try {
          // Try different recovery strategies
          const recovered = await this.recoveryStrategies.attemptRecovery(report);
          
          if (recovered) {
            console.log(`✅ Recovered stuck report: ${report.id}`);
            this.metrics.recoveryAttempts++;
            
            // Remove from stuck list
            await this.removeReportFromQueue(report.id);
            this.syncAttempts.delete(report.id);
          } else {
            // Move to manual review queue
            await this.moveToManualReview(report);
          }
        } catch (error) {
          console.error(`❌ Failed to recover report ${report.id}:`, error);
          await this.moveToManualReview(report);
        }
      }
    }
  
    async moveToManualReview(report) {
      try {
        let manualQueue = await idbKeyval.get('manual_review_queue') || [];
        manualQueue.push({
          ...report,
          reviewReason: 'recovery_failed',
          reviewTimestamp: new Date().toISOString(),
          attempts: this.syncAttempts.get(report.id) || 0
        });
        
        await idbKeyval.set('manual_review_queue', manualQueue);
        await this.removeReportFromQueue(report.id);
        
        this.showNotification(`⚠️ Report ${report.id.substring(0, 8)} needs manual review`, 'warning');
        this.updateManualReviewUI();
        
        console.log(`📋 Moved report to manual review: ${report.id}`);
      } catch (error) {
        console.error('❌ Failed to move report to manual review:', error);
      }
    }
  
    async recoverFailedSyncs() {
      if (!this.isOnline) return;
  
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        
        for (const report of queue) {
          const attempts = this.syncAttempts.get(report.id) || 0;
          
          if (attempts < this.config.maxSyncAttempts) {
            await this.attemptSyncWithBackoff(report);
          }
        }
      } catch (error) {
        console.error('❌ Error in recovery sync:', error);
      }
    }
  
    async attemptSyncWithBackoff(report) {
      const attempts = this.syncAttempts.get(report.id) || 0;
      const delay = Math.min(
        this.config.syncRetryDelay * Math.pow(this.config.backoffMultiplier, attempts),
        this.config.maxRetryDelay
      );
  
      setTimeout(async () => {
        try {
          console.log(`🔄 Retry sync attempt ${attempts + 1} for report: ${report.id}`);
          
          this.syncAttempts.set(report.id, attempts + 1);
          
          const success = await this.syncSingleReport(report);
          
          if (success) {
            console.log(`✅ Sync successful for report: ${report.id}`);
            this.metrics.syncSuccesses++;
            this.syncAttempts.delete(report.id);
            await this.removeReportFromQueue(report.id);
            this.updateSyncStatus();
          } else {
            console.warn(`⚠️ Sync failed for report: ${report.id}`);
            this.metrics.syncFailures++;
          }
        } catch (error) {
          console.error(`❌ Sync error for report ${report.id}:`, error);
          this.metrics.syncFailures++;
        }
      }, delay);
    }
  
    async syncSingleReport(report) {
      try {
        // Check for conflicts before syncing
        const conflict = await this.conflictResolver.checkForConflicts(report);
        
        if (conflict) {
          console.log(`⚠️ Conflict detected for report: ${report.id}`);
          const resolved = await this.conflictResolver.resolveConflict(report, conflict);
          
          if (!resolved) {
            await this.moveToManualReview(report);
            return false;
          }
          
          this.metrics.conflictsResolved++;
        }
        
        // Attempt sync
        const formData = new FormData();
        formData.append('json', JSON.stringify(report));
        
        if (report.has_offline_image) {
          const { blob } = await getImageBlob(report.id);
          if (blob) {
            formData.append('file', blob, report.image_metadata?.filename || 'image.jpg');
          }
        }
  
        const response = await fetch('/generate-report', {
          method: 'POST',
          body: formData,
          signal: AbortSignal.timeout(30000) // 30 second timeout
        });
  
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
  
        // Clean up on success
        if (report.has_offline_image) {
          await deleteImageBlob(report.id);
        }
  
        return true;
      } catch (error) {
        console.error('Sync error:', error);
        return false;
      }
    }
  
    async attemptFullRecovery() {
      console.log('🚀 Starting full recovery process...');
      
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        
        if (queue.length === 0) {
          console.log('✅ No reports to recover');
          return;
        }
  
        console.log(`🔄 Recovering ${queue.length} reports...`);
        
        // Show recovery progress
        this.showRecoveryProgress(0, queue.length);
        
        let successCount = 0;
        let failureCount = 0;
        
        for (let i = 0; i < queue.length; i++) {
          const report = queue[i];
          
          try {
            const success = await this.syncSingleReport(report);
            
            if (success) {
              successCount++;
              await this.removeReportFromQueue(report.id);
            } else {
              failureCount++;
            }
            
            // Update progress
            this.showRecoveryProgress(i + 1, queue.length);
            
            // Small delay to prevent overwhelming the server
            await new Promise(resolve => setTimeout(resolve, 500));
          } catch (error) {
            console.error(`Recovery error for report ${report.id}:`, error);
            failureCount++;
          }
        }
        
        // Show final results
        this.showRecoveryResults(successCount, failureCount);
        
      } catch (error) {
        console.error('❌ Full recovery failed:', error);
        this.showNotification('❌ Recovery process failed', 'error');
      }
    }
  
    async removeReportFromQueue(reportId) {
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        const updatedQueue = queue.filter(report => report.id !== reportId);
        await idbKeyval.set('sync_queue', updatedQueue);
        this.updateSyncStatus();
      } catch (error) {
        console.error('Error removing report from queue:', error);
      }
    }
  
    // UI Management
    setupUI() {
      this.createOfflineStatusPanel();
      this.createRecoveryControls();
      this.createManualReviewPanel();
      this.createStressTestControls();
    }
  
    createOfflineStatusPanel() {
      if (document.getElementById('offlineStatusPanel')) return;
  
      const panel = document.createElement('div');
      panel.id = 'offlineStatusPanel';
      panel.className = 'offline-status-panel';
      panel.innerHTML = `
        <div class="status-header">
          <span class="status-icon" id="networkStatusIcon">🟢</span>
          <span class="status-text" id="networkStatusText">Online</span>
          <button class="status-toggle" onclick="offlineRecovery.toggleNetworkSimulation()">
            📶 Simulate Offline
          </button>
        </div>
        <div class="status-details">
          <div class="status-metric">
            <span class="metric-label">Queue:</span>
            <span class="metric-value" id="queueCount">0</span>
          </div>
          <div class="status-metric">
            <span class="metric-label">Failed:</span>
            <span class="metric-value" id="failedCount">0</span>
          </div>
          <div class="status-metric">
            <span class="metric-label">Manual Review:</span>
            <span class="metric-value" id="manualReviewCount">0</span>
          </div>
        </div>
      `;
  
      // Add styles
      const style = document.createElement('style');
      style.textContent = `
        .offline-status-panel {
          position: fixed;
          top: 60px;
          right: 10px;
          background: rgba(0, 0, 0, 0.9);
          color: white;
          padding: 1rem;
          border-radius: 8px;
          font-family: monospace;
          font-size: 0.8rem;
          z-index: 9999;
          min-width: 200px;
        }
        .status-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.5rem;
        }
        .status-toggle {
          margin-left: auto;
          padding: 0.25rem 0.5rem;
          border: none;
          border-radius: 4px;
          background: #333;
          color: white;
          cursor: pointer;
          font-size: 0.7rem;
        }
        .status-details {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.5rem;
        }
        .status-metric {
          display: flex;
          justify-content: space-between;
        }
        .metric-value {
          font-weight: bold;
        }
        .recovery-controls {
          margin-top: 0.5rem;
          display: flex;
          gap: 0.5rem;
        }
        .recovery-btn {
          padding: 0.25rem 0.5rem;
          border: none;
          border-radius: 4px;
          background: #2563eb;
          color: white;
          cursor: pointer;
          font-size: 0.7rem;
        }
        .recovery-progress {
          margin-top: 0.5rem;
          padding: 0.5rem;
          background: #1f2937;
          border-radius: 4px;
          display: none;
        }
        .progress-bar {
          width: 100%;
          height: 4px;
          background: #374151;
          border-radius: 2px;
          overflow: hidden;
          margin: 0.25rem 0;
        }
        .progress-fill {
          height: 100%;
          background: #10b981;
          transition: width 0.3s ease;
        }
      `;
      document.head.appendChild(style);
      document.body.appendChild(panel);
    }
  
    createRecoveryControls() {
      const panel = document.getElementById('offlineStatusPanel');
      if (!panel) return;
  
      const controls = document.createElement('div');
      controls.className = 'recovery-controls';
      controls.innerHTML = `
        <button class="recovery-btn" onclick="offlineRecovery.attemptFullRecovery()">
          🔄 Recover All
        </button>
        <button class="recovery-btn" onclick="offlineRecovery.clearQueue()">
          🗑️ Clear Queue
        </button>
        <button class="recovery-btn" onclick="offlineRecovery.runStressTest()">
          🧪 Stress Test
        </button>
      `;
      panel.appendChild(controls);
  
      // Add progress indicator
      const progress = document.createElement('div');
      progress.id = 'recoveryProgress';
      progress.className = 'recovery-progress';
      progress.innerHTML = `
        <div class="progress-text">Recovery in progress...</div>
        <div class="progress-bar">
          <div class="progress-fill" id="progressFill"></div>
        </div>
        <div class="progress-details" id="progressDetails">0 / 0 reports</div>
      `;
      panel.appendChild(progress);
    }
  
    createManualReviewPanel() {
      const existingPanel = document.getElementById('manualReviewPanel');
      if (existingPanel) return;
  
      const panel = document.createElement('div');
      panel.id = 'manualReviewPanel';
      panel.style.cssText = `
        position: fixed;
        bottom: 10px;
        right: 10px;
        background: rgba(255, 193, 7, 0.95);
        color: black;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.8rem;
        z-index: 9998;
        max-width: 300px;
        display: none;
      `;
      panel.innerHTML = `
        <div><strong>⚠️ Manual Review Required</strong></div>
        <div id="manualReviewList"></div>
        <button onclick="offlineRecovery.showManualReviewDetails()" 
                style="margin-top: 0.5rem; padding: 0.25rem 0.5rem;">
          📋 View Details
        </button>
      `;
      document.body.appendChild(panel);
    }
  
    createStressTestControls() {
      // Stress test controls are added as part of recovery controls
    }
  
    // Network simulation for testing
    toggleNetworkSimulation() {
      if (this.networkSimulationActive) {
        this.stopNetworkSimulation();
      } else {
        this.startNetworkSimulation();
      }
    }
  
    startNetworkSimulation() {
      this.networkSimulationActive = true;
      this.originalFetch = window.fetch;
      
      // Override fetch to simulate network failures
      window.fetch = (...args) => {
        return Promise.reject(new Error('Simulated network failure'));
      };
      
      this.isOnline = false;
      this.onNetworkLost();
      
      document.getElementById('networkStatusIcon').textContent = '🔴';
      document.getElementById('networkStatusText').textContent = 'Offline (Simulated)';
      
      const toggle = document.querySelector('.status-toggle');
      toggle.textContent = '🟢 Restore Network';
      
      console.log('🔴 Network simulation started');
    }
  
    stopNetworkSimulation() {
      this.networkSimulationActive = false;
      
      if (this.originalFetch) {
        window.fetch = this.originalFetch;
      }
      
      this.isOnline = navigator.onLine;
      this.onNetworkRecovered();
      
      document.getElementById('networkStatusIcon').textContent = '🟢';
      document.getElementById('networkStatusText').textContent = 'Online';
      
      const toggle = document.querySelector('.status-toggle');
      toggle.textContent = '📶 Simulate Offline';
      
      console.log('🟢 Network simulation stopped');
    }
  
    // UI Updates
    updateNetworkStatus(online) {
      const icon = document.getElementById('networkStatusIcon');
      const text = document.getElementById('networkStatusText');
      
      if (icon && text) {
        icon.textContent = online ? '🟢' : '🔴';
        text.textContent = online ? 'Online' : 'Offline';
      }
    }
  
    async updateSyncStatus() {
      try {
        const queue = await idbKeyval.get('sync_queue') || [];
        const manualQueue = await idbKeyval.get('manual_review_queue') || [];
        
        const queueCount = document.getElementById('queueCount');
        const failedCount = document.getElementById('failedCount');
        const manualReviewCount = document.getElementById('manualReviewCount');
        
        if (queueCount) queueCount.textContent = queue.length;
        if (failedCount) failedCount.textContent = this.metrics.syncFailures;
        if (manualReviewCount) manualReviewCount.textContent = manualQueue.length;
        
        // Show/hide manual review panel
        const panel = document.getElementById('manualReviewPanel');
        if (panel) {
          panel.style.display = manualQueue.length > 0 ? 'block' : 'none';
        }
      } catch (error) {
        console.error('Error updating sync status:', error);
      }
    }
  
    updateManualReviewUI() {
      // Update the manual review panel with current items
      this.updateSyncStatus();
    }
  
    showRecoveryProgress(current, total) {
      const progress = document.getElementById('recoveryProgress');
      const fill = document.getElementById('progressFill');
      const details = document.getElementById('progressDetails');
      
      if (progress && fill && details) {
        progress.style.display = 'block';
        
        const percentage = total > 0 ? (current / total) * 100 : 0;
        fill.style.width = `${percentage}%`;
        details.textContent = `${current} / ${total} reports`;
        
        if (current === total) {
          setTimeout(() => {
            progress.style.display = 'none';
          }, 2000);
        }
      }
    }
  
    showRecoveryResults(successCount, failureCount) {
      const message = `Recovery complete: ${successCount} succeeded, ${failureCount} failed`;
      const type = failureCount === 0 ? 'success' : 'warning';
      this.showNotification(message, type);
    }
  
    showNotification(message, type = 'info') {
      // Create notification element
      const notification = document.createElement('div');
      notification.className = `notification notification-${type}`;
      notification.textContent = message;
      notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        z-index: 10000;
        animation: slideDown 0.3s ease;
        background: ${type === 'success' ? '#10b981' : 
                     type === 'warning' ? '#f59e0b' : 
                     type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
      `;
      
      // Add animation styles if not already added
      if (!document.getElementById('notificationStyles')) {
        const style = document.createElement('style');
        style.id = 'notificationStyles';
        style.textContent = `
          @keyframes slideDown {
            from { transform: translateX(-50%) translateY(-100%); opacity: 0; }
            to { transform: translateX(-50%) translateY(0); opacity: 1; }
          }
        `;
        document.head.appendChild(style);
      }
      
      document.body.appendChild(notification);
      
      // Remove after 5 seconds
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 5000);
    }
  
    // Utility methods
    async clearQueue() {
      if (confirm('Are you sure you want to clear the sync queue? This will delete all pending reports.')) {
        await idbKeyval.set('sync_queue', []);
        await idbKeyval.set('manual_review_queue', []);
        this.syncAttempts.clear();
        this.updateSyncStatus();
        this.showNotification('✅ Queue cleared', 'success');
      }
    }
  
    async runStressTest() {
      console.log('🧪 Starting stress test...');
      this.showNotification('🧪 Running offline stress test...', 'info');
      
      try {
        const results = await this.stressTestRunner.runFullTest();
        console.log('📊 Stress test results:', results);
        
        // Show results
        alert(`Stress Test Results:
          
  Reports Generated: ${results.reportsGenerated}
  Sync Success Rate: ${results.syncSuccessRate}%
  Average Recovery Time: ${results.avgRecoveryTime}ms
  Memory Usage: ${results.memoryUsage}
  Overall Score: ${results.overallScore}/100`);
        
      } catch (error) {
        console.error('❌ Stress test failed:', error);
        this.showNotification('❌ Stress test failed', 'error');
      }
    }
  
    showManualReviewDetails() {
      // Open detailed manual review interface
      window.open('/manual-review', '_blank');
    }
  
    // Metrics and monitoring
    updateRecoveryMetrics() {
      // Update internal metrics for monitoring
      this.metrics.lastRecoveryCheck = Date.now();
    }
  
    getMetrics() {
      return {
        ...this.metrics,
        queueSize: this.syncAttempts.size,
        configuredMaxAttempts: this.config.maxSyncAttempts,
        isOnline: this.isOnline
      };
    }
  
    async checkAndRecoverQueue() {
      // Initial check on startup
      await this.performRecoveryCheck();
      await this.updateSyncStatus();
    }
  
    assessConnectionQuality() {
      if (!('connection' in navigator)) return;
      
      const connection = navigator.connection;
      const quality = {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
        saveData: connection.saveData
      };
      
      console.log('📶 Connection quality:', quality);
      
      // Adjust sync strategy based on connection quality
      if (quality.effectiveType === 'slow-2g' || quality.downlink < 0.5) {
        this.config.syncRetryDelay = 5000; // Longer delays for slow connections
        this.config.maxRetryDelay = 60000;
      } else {
        this.config.syncRetryDelay = 2000; // Default delays
        this.config.maxRetryDelay = 30000;
      }
    }
  
    async cleanupOldData() {
      try {
        const cutoffTime = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days ago
        
        // Clean up old manual review items
        const manualQueue = await idbKeyval.get('manual_review_queue') || [];
        const filteredQueue = manualQueue.filter(item => {
          return new Date(item.reviewTimestamp).getTime() > cutoffTime;
        });
        
        if (filteredQueue.length !== manualQueue.length) {
          await idbKeyval.set('manual_review_queue', filteredQueue);
          console.log(`🧹 Cleaned up ${manualQueue.length - filteredQueue.length} old manual review items`);
        }
      } catch (error) {
        console.error('Error cleaning up old data:', error);
      }
    }
  }
  
  // Conflict Resolution System
  class ConflictResolver {
    async checkForConflicts(report) {
      // Simulate conflict detection
      // In production, this would check server state
      return null; // No conflicts for now
    }
  
    async resolveConflict(report, conflict) {
      // Automatic conflict resolution strategies
      console.log('🔧 Resolving conflict for report:', report.id);
      
      // Strategy 1: Timestamp-based resolution
      if (conflict.type === 'timestamp') {
        return this.resolveByTimestamp(report, conflict);
      }
      
      // Strategy 2: Severity-based resolution
      if (conflict.type === 'severity') {
        return this.resolveBySeverity(report, conflict);
      }
      
      // Default: require manual resolution
      return false;
    }
  
    resolveByTimestamp(report, conflict) {
      // Use most recent version
      return new Date(report.timestamp) > new Date(conflict.serverTimestamp);
    }
  
    resolveBySeverity(report, conflict) {
      // Use higher severity version
      return report.severity > conflict.serverSeverity;
    }
  }
  
  // Recovery Strategies
  class RecoveryStrategies {
    async attemptRecovery(report) {
      console.log(`🔧 Attempting recovery for report: ${report.id}`);
      
      // Strategy 1: Retry with simplified payload
      try {
        const simplified = this.createSimplifiedReport(report);
        const success = await this.syncSimplifiedReport(simplified);
        if (success) return true;
      } catch (error) {
        console.warn('Simplified sync failed:', error);
      }
      
      // Strategy 2: Split large reports
      if (this.isLargeReport(report)) {
        try {
          const success = await this.syncInParts(report);
          if (success) return true;
        } catch (error) {
          console.warn('Part-based sync failed:', error);
        }
      }
      
      // Strategy 3: Compress and retry
      try {
        const compressed = await this.compressReport(report);
        const success = await this.syncCompressedReport(compressed);
        if (success) return true;
      } catch (error) {
        console.warn('Compressed sync failed:', error);
      }
      
      return false;
    }
  
    createSimplifiedReport(report) {
      // Remove non-essential data
      return {
        id: report.id,
        timestamp: report.timestamp,
        location: report.location,
        coordinates: report.coordinates,
        severity: report.severity,
        notes: report.notes.substring(0, 500), // Truncate notes
        hazards: report.hazards.slice(0, 3), // Limit hazards
        // Remove: image_url, ai_analysis, detailed metadata
        simplified: true
      };
    }
  
    isLargeReport(report) {
      const reportSize = JSON.stringify(report).length;
      return reportSize > 50000; // 50KB threshold
    }
  
    async syncInParts(report) {
      // Split report into metadata and attachments
      const metadata = { ...report };
      delete metadata.image_url;
      delete metadata.ai_analysis;
      
      // Sync metadata first
      const metadataSuccess = await this.syncReportPart(metadata, 'metadata');
      if (!metadataSuccess) return false;
      
      // Sync attachments separately if they exist
      if (report.has_offline_image) {
        const imageSuccess = await this.syncReportPart(
          { id: report.id, type: 'image' }, 'image'
        );
        if (!imageSuccess) return false;
      }
      
      return true;
    }
  
    async syncReportPart(partData, partType) {
      try {
        const response = await fetch(`/sync-part/${partType}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(partData),
          signal: AbortSignal.timeout(15000)
        });
        
        return response.ok;
      } catch (error) {
        console.error(`Failed to sync ${partType}:`, error);
        return false;
      }
    }
  
    async compressReport(report) {
      // Simple compression by removing whitespace and redundant data
      const compressed = {
        ...report,
        notes: report.notes.replace(/\s+/g, ' ').trim(),
        compressed: true
      };
      
      // Remove empty fields
      Object.keys(compressed).forEach(key => {
        if (compressed[key] === '' || compressed[key] === null || compressed[key] === undefined) {
          delete compressed[key];
        }
      });
      
      return compressed;
    }
  
    async syncSimplifiedReport(report) {
      try {
        const response = await fetch('/sync-simplified', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(report),
          signal: AbortSignal.timeout(10000)
        });
        
        return response.ok;
      } catch (error) {
        console.error('Simplified sync failed:', error);
        return false;
      }
    }
  
    async syncCompressedReport(report) {
      try {
        const response = await fetch('/sync-compressed', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Content-Encoding': 'simplified'
          },
          body: JSON.stringify(report),
          signal: AbortSignal.timeout(10000)
        });
        
        return response.ok;
      } catch (error) {
        console.error('Compressed sync failed:', error);
        return false;
      }
    }
  }
  
  // Stress Testing Suite
  class StressTestRunner {
    constructor() {
      this.testConfig = {
        reportCount: 50,
        imageSize: 1024 * 1024, // 1MB
        networkFailureRate: 0.3,
        testDuration: 30000, // 30 seconds
        concurrentRequests: 5
      };
    }
  
    async runFullTest() {
      console.log('🧪 Starting comprehensive stress test...');
      
      const startTime = Date.now();
      const initialMemory = this.getMemoryUsage();
      
      const results = {
        startTime,
        reportsGenerated: 0,
        syncAttempts: 0,
        syncSuccesses: 0,
        syncFailures: 0,
        conflicts: 0,
        recoveries: 0,
        avgResponseTime: 0,
        memoryUsage: initialMemory,
        errors: []
      };
  
      try {
        // Test 1: Generate flood of reports
        await this.testReportFlood(results);
        
        // Test 2: Network instability simulation
        await this.testNetworkInstability(results);
        
        // Test 3: Storage capacity test
        await this.testStorageCapacity(results);
        
        // Test 4: Recovery mechanisms
        await this.testRecoveryMechanisms(results);
        
        // Test 5: Memory stress test
        await this.testMemoryStress(results);
        
      } catch (error) {
        results.errors.push(error.message);
        console.error('Stress test error:', error);
      }
  
      // Calculate final metrics
      results.endTime = Date.now();
      results.totalDuration = results.endTime - results.startTime;
      results.syncSuccessRate = results.syncAttempts > 0 ? 
        (results.syncSuccesses / results.syncAttempts) * 100 : 0;
      results.avgRecoveryTime = results.totalDuration / Math.max(results.recoveries, 1);
      results.finalMemory = this.getMemoryUsage();
      results.memoryIncrease = this.calculateMemoryIncrease(initialMemory, results.finalMemory);
      results.overallScore = this.calculateOverallScore(results);
  
      console.log('📊 Stress test completed:', results);
      return results;
    }
  
    async testReportFlood(results) {
      console.log('🌊 Testing report flood...');
      
      const promises = [];
      for (let i = 0; i < this.testConfig.reportCount; i++) {
        promises.push(this.generateTestReport(i));
        results.reportsGenerated++;
        
        // Batch requests to avoid overwhelming
        if (promises.length >= this.testConfig.concurrentRequests) {
          await Promise.allSettled(promises);
          promises.length = 0;
          await new Promise(resolve => setTimeout(resolve, 100)); // Small delay
        }
      }
      
      // Handle remaining promises
      if (promises.length > 0) {
        await Promise.allSettled(promises);
      }
    }
  
    async testNetworkInstability(results) {
      console.log('📶 Testing network instability...');
      
      // Simulate intermittent network failures
      const originalFetch = window.fetch;
      window.fetch = (...args) => {
        if (Math.random() < this.testConfig.networkFailureRate) {
          results.syncFailures++;
          return Promise.reject(new Error('Simulated network failure'));
        }
        results.syncAttempts++;
        return originalFetch(...args).then(response => {
          if (response.ok) results.syncSuccesses++;
          else results.syncFailures++;
          return response;
        }).catch(error => {
          results.syncFailures++;
          throw error;
        });
      };
      
      // Generate reports during network instability
      for (let i = 0; i < 20; i++) {
        await this.generateTestReport(`unstable_${i}`);
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      // Restore normal fetch
      window.fetch = originalFetch;
    }
  
    async testStorageCapacity(results) {
      console.log('💾 Testing storage capacity...');
      
      try {
        // Generate large reports to test storage limits
        const largeReports = [];
        for (let i = 0; i < 10; i++) {
          const largeReport = await this.generateLargeTestReport(i);
          largeReports.push(largeReport);
          
          // Try to store in IndexedDB
          try {
            await idbKeyval.set(`stress_test_${i}`, largeReport);
          } catch (error) {
            results.errors.push(`Storage capacity exceeded at report ${i}: ${error.message}`);
            break;
          }
        }
        
        // Clean up test data
        for (let i = 0; i < 10; i++) {
          await idbKeyval.del(`stress_test_${i}`);
        }
        
      } catch (error) {
        results.errors.push(`Storage test failed: ${error.message}`);
      }
    }
  
    async testRecoveryMechanisms(results) {
      console.log('🔧 Testing recovery mechanisms...');
      
      // Create stuck reports
      const stuckReports = [];
      for (let i = 0; i < 5; i++) {
        const stuckReport = await this.generateTestReport(`stuck_${i}`);
        stuckReport.timestamp = new Date(Date.now() - 600000).toISOString(); // 10 minutes ago
        stuckReports.push(stuckReport);
      }
      
      // Add to queue
      const currentQueue = await idbKeyval.get('sync_queue') || [];
      await idbKeyval.set('sync_queue', [...currentQueue, ...stuckReports]);
      
      // Test recovery
      const recoveryStart = Date.now();
      if (window.offlineRecovery) {
        await window.offlineRecovery.identifyStuckReports();
        results.recoveries += stuckReports.length;
      }
      const recoveryTime = Date.now() - recoveryStart;
      
      console.log(`Recovery test completed in ${recoveryTime}ms`);
    }
  
    async testMemoryStress(results) {
      console.log('🧠 Testing memory stress...');
      
      const memoryHogs = [];
      try {
        // Create memory-intensive operations
        for (let i = 0; i < 100; i++) {
          // Create large objects
          const largeObject = new Array(10000).fill().map(() => ({
            id: crypto.randomUUID(),
            data: new Array(1000).fill(Math.random()),
            timestamp: new Date().toISOString()
          }));
          memoryHogs.push(largeObject);
          
          // Check memory periodically
          if (i % 20 === 0) {
            const currentMemory = this.getMemoryUsage();
            console.log(`Memory check ${i}: ${currentMemory}`);
          }
        }
        
        // Force garbage collection if available
        if (window.gc) {
          window.gc();
        }
        
      } catch (error) {
        results.errors.push(`Memory stress test failed: ${error.message}`);
      } finally {
        // Clean up
        memoryHogs.length = 0;
      }
    }
  
    async generateTestReport(id) {
      const testReport = {
        id: `stress_test_${id}_${crypto.randomUUID()}`,
        timestamp: new Date().toISOString(),
        location: `Test Location ${id}`,
        coordinates: [30.2672 + (Math.random() - 0.5) * 0.1, -97.7431 + (Math.random() - 0.5) * 0.1],
        hazards: ['fire', 'structural_damage', 'debris'].slice(0, Math.floor(Math.random() * 3) + 1),
        severity: Math.floor(Math.random() * 10) + 1,
        notes: `Stress test report ${id}. ${new Array(100).fill('Test data').join(' ')}`,
        stress_test: true,
        ai_analysis: {
          confidence: Math.random(),
          processing_time: Math.random() * 1000,
          fake_data: new Array(50).fill().map(() => Math.random())
        }
      };
  
      // Add to queue
      const queue = await idbKeyval.get('sync_queue') || [];
      queue.push(testReport);
      await idbKeyval.set('sync_queue', queue);
  
      return testReport;
    }
  
    async generateLargeTestReport(id) {
      const baseReport = await this.generateTestReport(id);
      
      // Add large data structures
      baseReport.large_data = {
        coordinates_history: new Array(1000).fill().map(() => [
          Math.random() * 180 - 90,
          Math.random() * 360 - 180
        ]),
        sensor_readings: new Array(5000).fill().map(() => ({
          timestamp: Date.now(),
          temperature: Math.random() * 100,
          humidity: Math.random() * 100,
          pressure: Math.random() * 1013 + 900
        })),
        image_metadata: {
          large_binary_data: new Array(10000).fill(0).map(() => Math.floor(Math.random() * 256))
        }
      };
  
      return baseReport;
    }
  
    getMemoryUsage() {
      if (performance.memory) {
        return {
          used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
          total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
          limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
        };
      }
      return { used: 0, total: 0, limit: 0 };
    }
  
    calculateMemoryIncrease(initial, final) {
      if (typeof initial === 'object' && typeof final === 'object') {
        return final.used - initial.used;
      }
      return 0;
    }
  
    calculateOverallScore(results) {
      let score = 100;
      
      // Deduct points for failures
      if (results.syncSuccessRate < 90) score -= (90 - results.syncSuccessRate);
      if (results.errors.length > 0) score -= results.errors.length * 5;
      if (results.memoryIncrease > 50) score -= 10; // Penalize memory leaks
      if (results.avgRecoveryTime > 5000) score -= 10; // Penalize slow recovery
      
      return Math.max(0, Math.round(score));
    }
  }
  
  // Initialize the offline recovery system
  window.offlineRecovery = new OfflineRecoveryManager();
  
  // Global utilities for integration
  window.OfflineStressTesting = {
    runStressTest: () => window.offlineRecovery.runStressTest(),
    simulateNetworkFailure: () => window.offlineRecovery.startNetworkSimulation(),
    restoreNetwork: () => window.offlineRecovery.stopNetworkSimulation(),
    clearAllQueues: () => window.offlineRecovery.clearQueue(),
    getMetrics: () => window.offlineRecovery.getMetrics(),
    forceRecovery: () => window.offlineRecovery.attemptFullRecovery()
  };
  
  console.log('📶 Offline Recovery & Stress Testing System loaded!');
  console.log('🧪 Use OfflineStressTesting.runStressTest() to start comprehensive testing');
  console.log('📊 Use OfflineStressTesting.getMetrics() to view current metrics');