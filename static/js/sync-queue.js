// Enhanced sync-queue.js - Emergency Data Synchronization System
import { openDB } from '/static/js/idb.mjs';

// Configuration
const CONFIG = {
  DB_NAME: 'emergency_sync_queue',
  DB_VERSION: 2,
  STORES: {
    REPORTS: 'queued_reports',
    METADATA: 'sync_metadata',
    FAILURES: 'sync_failures'
  },
  SYNC_INTERVAL: 30000, // 30 seconds
  MAX_RETRIES: 5,
  RETRY_DELAYS: [1000, 2000, 5000, 10000, 30000], // Exponential backoff
  PRIORITY_SYNC_TYPES: ['emergency_report', 'triage_assessment', 'emergency_broadcast'],
  MAX_QUEUE_SIZE: 1000,
  BATCH_SIZE: 10
};

// Global state
let syncInterval = null;
let statusCallback = null;
let isCurrentlySyncing = false;
let lastSyncAttempt = null;
let syncStats = {
  totalSynced: 0,
  totalFailed: 0,
  lastSuccessfulSync: null
};

/**
 * Initialize the database with enhanced schema
 */
async function initializeDB() {
  try {
    const db = await openDB(CONFIG.DB_NAME, CONFIG.DB_VERSION, {
      upgrade(db, oldVersion, newVersion, transaction) {
        console.log(`📊 Upgrading sync queue DB from v${oldVersion} to v${newVersion}`);
        
        // Reports store
        if (!db.objectStoreNames.contains(CONFIG.STORES.REPORTS)) {
          const reportsStore = db.createObjectStore(CONFIG.STORES.REPORTS, { 
            keyPath: 'id',
            autoIncrement: false 
          });
          reportsStore.createIndex('timestamp', 'timestamp');
          reportsStore.createIndex('priority', 'priority');
          reportsStore.createIndex('type', 'type');
          reportsStore.createIndex('retryCount', 'retryCount');
        }
        
        // Metadata store
        if (!db.objectStoreNames.contains(CONFIG.STORES.METADATA)) {
          db.createObjectStore(CONFIG.STORES.METADATA, { keyPath: 'key' });
        }
        
        // Failures store
        if (!db.objectStoreNames.contains(CONFIG.STORES.FAILURES)) {
          const failuresStore = db.createObjectStore(CONFIG.STORES.FAILURES, { 
            keyPath: 'id',
            autoIncrement: true 
          });
          failuresStore.createIndex('originalId', 'originalId');
          failuresStore.createIndex('timestamp', 'timestamp');
        }
      },
    });
    
    // Initialize metadata if needed
    await initializeMetadata(db);
    
    return db;
  } catch (error) {
    console.error('❌ Failed to initialize sync queue database:', error);
    throw error;
  }
}

/**
 * Initialize metadata store with default values
 */
async function initializeMetadata(db) {
  const tx = db.transaction([CONFIG.STORES.METADATA], 'readwrite');
  const store = tx.objectStore(CONFIG.STORES.METADATA);
  
  const defaultMetadata = [
    { key: 'syncStats', value: syncStats },
    { key: 'lastCleanup', value: Date.now() },
    { key: 'queueVersion', value: CONFIG.DB_VERSION }
  ];
  
  for (const item of defaultMetadata) {
    const existing = await store.get(item.key);
    if (!existing) {
      await store.put(item);
    }
  }
  
  await tx.complete;
}

/**
 * Enhanced report queuing with priority and validation
 */
export async function queueReport(data, options = {}) {
  try {
    const db = await initializeDB();
    
    // Validate required data
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid report data provided');
    }
    
    // Check queue size limit
    const currentSize = await getQueueLength();
    if (currentSize >= CONFIG.MAX_QUEUE_SIZE) {
      console.warn('⚠️ Queue size limit reached, removing oldest non-priority items');
      await cleanupOldEntries(db);
    }
    
    // Create enhanced report object
    const queuedReport = {
      id: options.id || generateReportId(),
      data: data,
      type: options.type || 'general_report',
      priority: options.priority || getPriorityFromType(options.type),
      timestamp: Date.now(),
      retryCount: 0,
      lastAttempt: null,
      endpoint: options.endpoint || '/generate-report',
      metadata: {
        userAgent: navigator.userAgent,
        online: navigator.onLine,
        source: options.source || 'unknown',
        ...options.metadata
      }
    };
    
    // Store in database
    const tx = db.transaction([CONFIG.STORES.REPORTS], 'readwrite');
    await tx.objectStore(CONFIG.STORES.REPORTS).put(queuedReport);
    await tx.complete;
    
    console.log(`📝 Report queued: ${queuedReport.id} (${queuedReport.type}, priority: ${queuedReport.priority})`);
    
    // Update status
    await updateSyncStatus();
    
    // Try immediate sync if high priority and online
    if (queuedReport.priority === 'critical' && navigator.onLine) {
      setTimeout(trySyncQueuedReports, 100);
    }
    
    return queuedReport.id;
    
  } catch (error) {
    console.error('❌ Failed to queue report:', error);
    throw error;
  }
}

/**
 * Get queued reports with enhanced filtering
 */
export async function getQueuedReports(options = {}) {
  try {
    const db = await initializeDB();
    const tx = db.transaction([CONFIG.STORES.REPORTS], 'readonly');
    const store = tx.objectStore(CONFIG.STORES.REPORTS);
    
    let reports;
    
    if (options.priority) {
      const index = store.index('priority');
      reports = await index.getAll(options.priority);
    } else if (options.type) {
      const index = store.index('type');
      reports = await index.getAll(options.type);
    } else {
      reports = await store.getAll();
    }
    
    await tx.complete;
    
    // Sort by priority and timestamp
    reports.sort((a, b) => {
      const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
      const aPriority = priorityOrder[a.priority] || 3;
      const bPriority = priorityOrder[b.priority] || 3;
      
      if (aPriority !== bPriority) {
        return aPriority - bPriority;
      }
      
      return a.timestamp - b.timestamp;
    });
    
    return options.limit ? reports.slice(0, options.limit) : reports;
    
  } catch (error) {
    console.error('❌ Failed to get queued reports:', error);
    return [];
  }
}

/**
 * Remove a queued report
 */
export async function removeQueuedReport(reportId) {
  try {
    const db = await initializeDB();
    const tx = db.transaction([CONFIG.STORES.REPORTS], 'readwrite');
    await tx.objectStore(CONFIG.STORES.REPORTS).delete(reportId);
    await tx.complete;
    
    console.log(`🗑️ Removed queued report: ${reportId}`);
    await updateSyncStatus();
    
  } catch (error) {
    console.error('❌ Failed to remove queued report:', error);
  }
}

/**
 * Get current queue statistics
 */
export async function getQueueLength() {
  try {
    const db = await initializeDB();
    const tx = db.transaction([CONFIG.STORES.REPORTS], 'readonly');
    const count = await tx.objectStore(CONFIG.STORES.REPORTS).count();
    await tx.complete;
    return count;
  } catch (error) {
    console.error('❌ Failed to get queue length:', error);
    return 0;
  }
}

/**
 * Get detailed queue statistics
 */
export async function getQueueStats() {
  try {
    const db = await initializeDB();
    const reports = await getQueuedReports();
    
    const stats = {
      total: reports.length,
      byPriority: {
        critical: reports.filter(r => r.priority === 'critical').length,
        high: reports.filter(r => r.priority === 'high').length,
        medium: reports.filter(r => r.priority === 'medium').length,
        low: reports.filter(r => r.priority === 'low').length
      },
      byType: {},
      oldestTimestamp: reports.length > 0 ? Math.min(...reports.map(r => r.timestamp)) : null,
      newestTimestamp: reports.length > 0 ? Math.max(...reports.map(r => r.timestamp)) : null,
      failedRetries: reports.filter(r => r.retryCount > 0).length
    };
    
    // Count by type
    reports.forEach(report => {
      stats.byType[report.type] = (stats.byType[report.type] || 0) + 1;
    });
    
    return stats;
    
  } catch (error) {
    console.error('❌ Failed to get queue stats:', error);
    return null;
  }
}

/**
 * Set status callback for UI updates
 */
export function setSyncStatusCallback(callback) {
  statusCallback = callback;
  // Immediately call with current status
  updateSyncStatus();
}

/**
 * Enhanced sync function with batch processing and retry logic
 */
export async function trySyncQueuedReports() {
  if (isCurrentlySyncing) {
    console.log('🔄 Sync already in progress, skipping...');
    return;
  }
  
  if (!navigator.onLine) {
    console.log('📴 Offline, skipping sync');
    return;
  }
  
  isCurrentlySyncing = true;
  lastSyncAttempt = Date.now();
  
  try {
    await updateSyncStatus(true);
    
    const db = await initializeDB();
    const reports = await getQueuedReports({ limit: CONFIG.BATCH_SIZE });
    
    if (reports.length === 0) {
      console.log('✅ Sync queue is empty');
      return;
    }
    
    console.log(`🔄 Starting sync for ${reports.length} reports`);
    
    let syncedCount = 0;
    let failedCount = 0;
    
    for (const report of reports) {
      try {
        const success = await syncSingleReport(db, report);
        if (success) {
          syncedCount++;
          syncStats.totalSynced++;
          syncStats.lastSuccessfulSync = Date.now();
        } else {
          failedCount++;
          syncStats.totalFailed++;
        }
      } catch (error) {
        console.error(`❌ Failed to sync report ${report.id}:`, error);
        failedCount++;
        syncStats.totalFailed++;
      }
    }
    
    console.log(`✅ Sync completed: ${syncedCount} synced, ${failedCount} failed`);
    
    // Update metadata
    await updateMetadata(db, 'syncStats', syncStats);
    
  } catch (error) {
    console.error('❌ Sync process failed:', error);
  } finally {
    isCurrentlySyncing = false;
    await updateSyncStatus();
  }
}

/**
 * Sync a single report with enhanced error handling
 */
async function syncSingleReport(db, report) {
  try {
    // Prepare request data
    const requestData = {
      ...report.data,
      _sync_metadata: {
        queueId: report.id,
        timestamp: report.timestamp,
        retryCount: report.retryCount,
        type: report.type,
        priority: report.priority
      }
    };
    
    // Make the request
    const response = await fetch(report.endpoint, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-Sync-Queue-ID': report.id,
        'X-Report-Type': report.type,
        'X-Priority': report.priority
      },
      body: JSON.stringify(requestData),
    });
    
    if (response.ok) {
      // Success - remove from queue
      await removeQueuedReport(report.id);
      console.log(`✅ Successfully synced report: ${report.id}`);
      return true;
      
    } else {
      // Server error - handle retry logic
      console.warn(`⚠️ Server rejected report ${report.id}: ${response.status} ${response.statusText}`);
      
      if (response.status >= 400 && response.status < 500) {
        // Client error - don't retry, move to failures
        await moveToFailures(db, report, `Client error: ${response.status}`);
        await removeQueuedReport(report.id);
      } else {
        // Server error - retry with backoff
        await handleRetry(db, report);
      }
      
      return false;
    }
    
  } catch (error) {
    console.warn(`⚠️ Network error syncing report ${report.id}:`, error.message);
    await handleRetry(db, report);
    return false;
  }
}

/**
 * Handle retry logic with exponential backoff
 */
async function handleRetry(db, report) {
  const newRetryCount = report.retryCount + 1;
  
  if (newRetryCount >= CONFIG.MAX_RETRIES) {
    console.warn(`❌ Max retries exceeded for report ${report.id}, moving to failures`);
    await moveToFailures(db, report, 'Max retries exceeded');
    await removeQueuedReport(report.id);
    return;
  }
  
  // Update retry count and schedule next attempt
  const updatedReport = {
    ...report,
    retryCount: newRetryCount,
    lastAttempt: Date.now(),
    nextAttempt: Date.now() + (CONFIG.RETRY_DELAYS[newRetryCount - 1] || 30000)
  };
  
  const tx = db.transaction([CONFIG.STORES.REPORTS], 'readwrite');
  await tx.objectStore(CONFIG.STORES.REPORTS).put(updatedReport);
  await tx.complete;
  
  console.log(`🔄 Scheduled retry ${newRetryCount}/${CONFIG.MAX_RETRIES} for report ${report.id}`);
}

/**
 * Move failed reports to failures store for analysis
 */
async function moveToFailures(db, report, reason) {
  try {
    const failureRecord = {
      originalId: report.id,
      data: report.data,
      type: report.type,
      priority: report.priority,
      timestamp: Date.now(),
      originalTimestamp: report.timestamp,
      retryCount: report.retryCount,
      failureReason: reason,
      metadata: report.metadata
    };
    
    const tx = db.transaction([CONFIG.STORES.FAILURES], 'readwrite');
    await tx.objectStore(CONFIG.STORES.FAILURES).add(failureRecord);
    await tx.complete;
    
    console.log(`📋 Moved failed report to failures store: ${report.id}`);
  } catch (error) {
    console.error('❌ Failed to move report to failures store:', error);
  }
}

/**
 * Start automatic retry with enhanced scheduling
 */
export function startAutoRetry(intervalMs = CONFIG.SYNC_INTERVAL) {
  if (syncInterval) {
    clearInterval(syncInterval);
  }
  
  syncInterval = setInterval(() => {
    if (navigator.onLine && !isCurrentlySyncing) {
      trySyncQueuedReports();
    }
  }, intervalMs);
  
  console.log(`⏰ Auto-retry started with ${intervalMs}ms interval`);
  
  // Also listen for online events
  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);
}

/**
 * Stop automatic retry
 */
export function stopAutoRetry() {
  if (syncInterval) {
    clearInterval(syncInterval);
    syncInterval = null;
  }
  
  window.removeEventListener('online', handleOnline);
  window.removeEventListener('offline', handleOffline);
  
  console.log('⏹️ Auto-retry stopped');
}

/**
 * Force immediate sync for critical reports
 */
export async function forceSyncCritical() {
  console.log('🚨 Force syncing critical reports');
  
  const criticalReports = await getQueuedReports({ priority: 'critical' });
  
  if (criticalReports.length === 0) {
    console.log('✅ No critical reports to sync');
    return;
  }
  
  // Sync critical reports immediately, even if already syncing
  const originalFlag = isCurrentlySyncing;
  isCurrentlySyncing = false;
  
  try {
    await trySyncQueuedReports();
  } finally {
    isCurrentlySyncing = originalFlag;
  }
}

/**
 * Clean up old entries and failures
 */
async function cleanupOldEntries(db) {
  try {
    const cutoffTime = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days
    
    // Clean up old non-priority reports
    const reports = await getQueuedReports();
    const toDelete = reports
      .filter(r => r.timestamp < cutoffTime && r.priority !== 'critical')
      .slice(0, Math.floor(CONFIG.MAX_QUEUE_SIZE * 0.1)); // Remove 10% of queue
    
    for (const report of toDelete) {
      await removeQueuedReport(report.id);
    }
    
    // Clean up old failures (keep last 100)
    const tx = db.transaction([CONFIG.STORES.FAILURES], 'readwrite');
    const failures = await tx.objectStore(CONFIG.STORES.FAILURES).getAll();
    
    if (failures.length > 100) {
      failures.sort((a, b) => b.timestamp - a.timestamp);
      const toDeleteFailures = failures.slice(100);
      
      for (const failure of toDeleteFailures) {
        await tx.objectStore(CONFIG.STORES.FAILURES).delete(failure.id);
      }
    }
    
    await tx.complete;
    
    console.log(`🧹 Cleaned up ${toDelete.length} old reports and ${failures.length > 100 ? failures.length - 100 : 0} old failures`);
    
  } catch (error) {
    console.error('❌ Cleanup failed:', error);
  }
}

/**
 * Update sync status and notify callbacks
 */
async function updateSyncStatus(syncing = false) {
  try {
    const queueLength = await getQueueLength();
    const stats = await getQueueStats();
    
    const status = {
      syncing: syncing || isCurrentlySyncing,
      queued: queueLength,
      stats: stats,
      lastSync: lastSyncAttempt,
      online: navigator.onLine
    };
    
    if (statusCallback) {
      statusCallback(status);
    }
    
  } catch (error) {
    console.error('❌ Failed to update sync status:', error);
  }
}

/**
 * Update metadata in database
 */
async function updateMetadata(db, key, value) {
  try {
    const tx = db.transaction([CONFIG.STORES.METADATA], 'readwrite');
    await tx.objectStore(CONFIG.STORES.METADATA).put({ key, value });
    await tx.complete;
  } catch (error) {
    console.error('❌ Failed to update metadata:', error);
  }
}

/**
 * Event handlers
 */
function handleOnline() {
  console.log('🌐 Connection restored - attempting sync');
  updateSyncStatus();
  
  // Immediate sync attempt
  setTimeout(trySyncQueuedReports, 1000);
}

function handleOffline() {
  console.log('📴 Connection lost');
  updateSyncStatus();
}

/**
 * Utility functions
 */
function generateReportId() {
  return `report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function getPriorityFromType(type) {
  if (CONFIG.PRIORITY_SYNC_TYPES.includes(type)) {
    return 'critical';
  }
  
  const priorityMap = {
    'emergency_report': 'critical',
    'triage_assessment': 'critical',
    'emergency_broadcast': 'critical',
    'crowd_report': 'high',
    'medical_report': 'high',
    'weather_report': 'medium',
    'general_report': 'low'
  };
  
  return priorityMap[type] || 'medium';
}

/**
 * Export additional utility functions
 */
export async function getFailedReports() {
  try {
    const db = await initializeDB();
    const tx = db.transaction([CONFIG.STORES.FAILURES], 'readonly');
    const failures = await tx.objectStore(CONFIG.STORES.FAILURES).getAll();
    await tx.complete;
    return failures;
  } catch (error) {
    console.error('❌ Failed to get failed reports:', error);
    return [];
  }
}

export async function retryFailedReport(failureId) {
  try {
    const db = await initializeDB();
    
    // Get the failed report
    const tx1 = db.transaction([CONFIG.STORES.FAILURES], 'readonly');
    const failure = await tx1.objectStore(CONFIG.STORES.FAILURES).get(failureId);
    await tx1.complete;
    
    if (!failure) {
      throw new Error('Failed report not found');
    }
    
    // Re-queue the report
    await queueReport(failure.data, {
      type: failure.type,
      priority: failure.priority,
      source: 'retry_failed'
    });
    
    // Remove from failures
    const tx2 = db.transaction([CONFIG.STORES.FAILURES], 'readwrite');
    await tx2.objectStore(CONFIG.STORES.FAILURES).delete(failureId);
    await tx2.complete;
    
    console.log(`🔄 Retrying failed report: ${failureId}`);
    
  } catch (error) {
    console.error('❌ Failed to retry failed report:', error);
  }
}

export function getSyncStats() {
  return { ...syncStats };
}

export function isOnline() {
  return navigator.onLine;
}

export function isSyncing() {
  return isCurrentlySyncing;
}

// Initialize on import
console.log('🚀 Enhanced sync queue system initialized');

// Auto-start if browser supports service workers
if ('serviceWorker' in navigator) {
  startAutoRetry();
}