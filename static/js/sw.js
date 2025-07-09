// ================================================================================
// 📋 CONFIGURATION & CONSTANTS
// ================================================================================

const SW_CONFIG = {
  VERSION: "v2.3.0",
  DB_NAME: "EmergencyResponseDB",
  DB_VERSION: 4, // Increment if schema changes
  NETWORK_TIMEOUT: 5000, // 5 seconds for network requests
  CACHE_TIMEOUT: 1000 // 1 second for cache-first strategy before network
};

const CACHE_NAMES = {
  CORE_APP: `disaster-assistant-app-${SW_CONFIG.VERSION}`,
  DATA_API: `disaster-assistant-data-${SW_CONFIG.VERSION}`,
  DYNAMIC_ASSETS: `disaster-assistant-dynamic-${SW_CONFIG.VERSION}`,
  IMAGES: `disaster-assistant-images-${SW_CONFIG.VERSION}`,
  PAGES: `disaster-assistant-pages-${SW_CONFIG.VERSION}`,
  // Gemma 3N specific caches
  GEMMA_VOICE: `gemma3n-voice-cache-${SW_CONFIG.VERSION}`,
  GEMMA_MULTIMODAL: `gemma3n-multimodal-cache-${SW_CONFIG.VERSION}`,
  GEMMA_MODELS: `gemma3n-models-cache-${SW_CONFIG.VERSION}`
};

const OFFLINE_URL = "/offline.html";

// Cache Size Limits
const CACHE_LIMITS = {
  DYNAMIC: 50, // Max items for dynamically cached pages/assets
  API: 100, // Max items for cached API responses
  IMAGES: 30, // Max items for cached images
  PAGES: 20, // Max items for critical pages
  GEMMA_VOICE: 50 * 1024 * 1024, // 50MB
  GEMMA_MULTIMODAL: 100 * 1024 * 1024, // 100MB
  GEMMA_MODELS: 200 * 1024 * 1024 // 200MB
};

// IndexedDB Store Configurations
const DB_STORES = {
  // Queue Stores for background sync
  BROADCAST_QUEUE: "broadcastQueue",
  CROWD_REPORT_QUEUE: "crowdReportQueue",
  TRIAGE_QUEUE: "triageQueue",
  EMERGENCY_REPORT_QUEUE: "emergencyReportQueue",
  WEATHER_RISK_QUEUE: "weatherRiskQueue",
  IMAGE_ANALYSIS_QUEUE: "imageAnalysisQueue",
  FEEDBACK_QUEUE: "feedbackQueue",
  EXPORT_REQUEST_QUEUE: "exportRequestQueue",
  GEMMA_VOICE_SYNC_QUEUE: "gemmaVoiceSyncQueue", // For Gemma 3N voice reports
  GEMMA_MULTIMODAL_SYNC_QUEUE: "gemmaMultimodalSyncQueue", // For Gemma 3N multimodal reports
  
  // Data Stores for persistent offline data
  SYNC_METADATA: "syncMetadata",
  OFFLINE_ANALYTICS: "offlineAnalytics",
  USER_PREFERENCES: "userPreferences",
  DEVICE_STATUS_LOG: "deviceStatusLog",
  SYNC_HISTORY_LOG: "syncHistoryLog",
  ARCHIVE_CACHE: "reportArchiveCache",
  ONBOARDING_STATE: "onboardingProgress",
  HELP_CACHE: "helpContentCache",
  ADMIN_CACHE: "adminDashboardCache",
  GEMMA_CONTEXT_CACHE: "gemmaContextCache" // For 128K context intelligence
};

// Background Sync Tags (must match event.tag in client-side registration)
const SYNC_TAGS = {
  BROADCAST: "emergency-broadcast-sync",
  CROWD_REPORT: "crowd-report-sync",
  TRIAGE: "triage-assessment-sync",
  EMERGENCY_REPORT: "emergency-report-sync",
  WEATHER_RISK: "weather-risk-sync",
  IMAGE_ANALYSIS: "image-analysis-sync",
  FEEDBACK: "feedback-sync",
  DEVICE_STATUS: "device-status-sync",
  EXPORT_REQUESTS: "export-requests-sync",
  GEMMA_VOICE: "gemma3n-voice-sync",
  GEMMA_MULTIMODAL: "gemma3n-multimodal-sync",
  ADMIN_ACTIONS: "admin-actions-sync",
  USER_PREFERENCES: "user-preferences-sync"
};

// URLs to pre-cache on install (critical for offline first)
const PRECACHE_URLS = [
  // Core Application Pages
  '/',
  OFFLINE_URL,
  '/manifest.json',
  '/static/css/styles.css', // Ensure base styles are caught
  
  // Main Navigation Pages
  '/hazards', '/generate', '/live-generate', '/map-reports', '/submit-report',
  '/predict', '/triage-form', '/test-offline',
  
  // Dashboard Pages
  '/admin', '/analytics', '/admin-dashboard',
  
  // Crowd Reports
  '/view-reports', '/submit-crowd-report', '/crowd-reports',
  
  // Patient Management
  '/triage', '/patients', '/patient-list', '/triage-dashboard',
  
  // System Management
  '/sync-status', '/device-status', '/report-archive', '/onboarding',
  '/feedback', '/help', '/reports/export',
  
  // 🆕 GEMMA 3N ENHANCED PAGES
  '/voice-emergency-reporter', '/multimodal-damage-assessment',
  '/context-intelligence-dashboard', '/adaptive-ai-settings',
  
  // Essential Static Assets & JS modules
  '/static/js/main.js', '/static/js/offline.js', '/static/js/idb.mjs',
  '/static/js/sync-manager.js', '/static/js/device-monitor.js',
  '/static/js/weather-risk.js', '/static/js/sync-queue.js', '/static/js/edge-ai.js',
  '/static/js/workers/broadcast.js', '/static/js/p2p/fallback-webrtc.js',
  '/static/js/triage-utils.js', '/static/js/pdf-export.js', '/static/js/patient-management.js',
  '/static/js/map-utils.js', '/static/js/archive-manager.js', '/static/js/onboarding-flow.js',
  '/static/js/admin-controls.js', '/static/js/feedback-handler.js', '/static/js/help-system.js',
  
  // Gemma 3N Specific JS/CSS (example paths)
  '/static/css/gemma3n.css',
  '/static/js/gemma3n-voice.js',
  '/static/js/gemma3n-multimodal.js',
  '/static/js/gemma3n-context.js',
  '/static/js/device-performance.js',
  
  // Icons and Images (example paths, adjust as needed)
  '/static/icons/icon-72x72.png', '/static/icons/icon-96x96.png',
  '/static/icons/icon-128x128.png', '/static/icons/icon-144x144.png',
  '/static/icons/icon-152x152.png', '/static/icons/icon-192x192.png',
  '/static/icons/icon-384x384.png', '/static/icons/icon-512x512.png',
  '/static/images/logo.png', '/static/images/emergency-icon.png',
  '/static/images/gemma3n-logo.png',
  
  // External Libraries (if hosted locally or via reliable CDN)
  "https://unpkg.com/leaflet/dist/leaflet.css",
  "https://unpkg.com/leaflet/dist/leaflet.js"
];

// API Endpoints that might be cached or require specific offline handling
const API_ENDPOINTS_TO_HANDLE = [
  // Core Data APIs (GET requests will be cached)
  '/api/crowd-report-locations', '/api/community-stats', '/api/recent-reports',
  '/api/triage-stats', '/api/dashboard-stats', '/api/map-statistics',
  '/api/demo-status', '/api/system-status', '/api/sync-queue', '/health',
  
  // Form Submission APIs (POST requests will be queued)
  '/api/submit-crowd-report', '/api/submit-emergency-report', '/submit-triage',
  '/predict-risk', '/analyze-sentiment', '/generate-report', '/broadcast',
  '/api/feedback', '/api/user-preferences', '/api/onboarding-progress',
  
  // Gemma 3N APIs (some GETs cached, POSTs queued/network-only)
  '/api/submit-voice-emergency-report', '/api/submit-damage-assessment',
  '/api/context-analysis', '/api/ai-model-status', '/api/optimize-ai-settings',
  '/api/device-performance', '/api/hazard-analysis',
  
  // Explicitly handled for queuing if offline
  '/api/patients', // For new patient submissions
  '/patients/', // For updates/discharges (will be queued if POST/PATCH)
];

// ================================================================================
// 🚀 SERVICE WORKER EVENT LISTENERS
// ================================================================================

// Installation Event
self.addEventListener("install", (event) => {
  console.log(`🔧 Service Worker ${SW_CONFIG.VERSION} installing...`);
  event.waitUntil(
    Promise.all([
      cacheManager.cacheInitialAssets(),
      databaseManager.initialize()
    ]).then(() => {
      console.log("✅ Service Worker installation complete");
      self.skipWaiting(); // Force activate immediately
    }).catch(error => {
      console.error("❌ Service Worker installation failed:", error);
    })
  );
});

// Activation Event
self.addEventListener("activate", (event) => {
  console.log(`🚀 Service Worker ${SW_CONFIG.VERSION} activating...`);
  event.waitUntil(
    Promise.all([
      cacheManager.cleanupOldCaches(),
      databaseManager.updateSchema(), // Ensure DB schema is up-to-date
      offlineManager.setupCapabilities(), // Setup periodic tasks, network monitoring etc.
      syncManager.checkPendingSyncs() // Attempt to sync any existing queued items
    ]).then(() => {
      console.log("✅ Service Worker activation complete");
      self.clients.claim(); // Take control of all open clients
      messageManager.notifyClientsOfOfflineReadiness();
    }).catch(error => {
      console.error("❌ Service Worker activation failed:", error);
    })
  );
});

// Fetch Event - Main Request Handler
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Ignore non-http(s) requests (e.g., chrome-extension://)
  if (!request.url.startsWith('http')) return;

  // Prioritize network for most POST requests that aren't specifically queued
  // and for non-GET API calls that might modify data on the server.
  if (request.method === "POST" || request.method === "PUT" || request.method === "DELETE" || request.method === "PATCH") {
    if (API_ENDPOINTS_TO_HANDLE.some(endpoint => url.pathname.includes(endpoint))) {
      event.respondWith(requestHandler.handleDataModificationRequest(request));
    } else {
      event.respondWith(requestHandler.handleDefaultNetworkOnly(request));
    }
  } else if (request.method === "GET") {
    if (url.pathname.startsWith("/static/")) {
      event.respondWith(requestHandler.handleStaticAsset(request));
    } else if (url.pathname.startsWith("/api/")) {
      event.respondWith(requestHandler.handleApiRequest(request));
    } else if (request.mode === 'navigate') { // Main page navigation
      event.respondWith(requestHandler.handlePageNavigation(request));
    } else { // Generic GET requests (e.g., other assets, data)
      event.respondWith(requestHandler.handleGenericGet(request));
    }
  } else {
    // Fallback for other HTTP methods
    event.respondWith(requestHandler.handleDefaultNetworkOnly(request));
  }
});

// Background Sync Event
self.addEventListener('sync', (event) => {
  console.log(`🔄 Background sync triggered: ${event.tag}`);
  const handler = syncManager.getSyncHandler(event.tag);
  if (handler) {
    event.waitUntil(handler());
  } else {
    console.warn(`⚠️ Unknown sync tag: ${event.tag}`);
  }
});

// Message Event (for communication from client to SW)
self.addEventListener("message", (event) => {
  messageManager.handleMessage(event);
});

// Push Notification Events
self.addEventListener("push", (event) => {
  notificationManager.handlePushNotification(event);
});

self.addEventListener("notificationclick", (event) => {
  notificationManager.handleNotificationClick(event);
});

// Error Handling for Service Worker's own operations
self.addEventListener('error', (event) => {
  errorManager.logServiceWorkerError(event.error);
});

self.addEventListener('unhandledrejection', (event) => {
  errorManager.logServiceWorkerError(event.reason);
});

// ================================================================================
// 📦 CACHE MANAGER
// ================================================================================

const cacheManager = {
  async cacheInitialAssets() {
    console.log("📦 Caching core application files and critical pages...");
    const cache = await caches.open(CACHE_NAMES.CORE_APP);
    // Add all critical assets and pages to the main app cache
    await cache.addAll(PRECACHE_URLS);
    console.log(`✅ ${PRECACHE_URLS.length} essential files cached.`);

    // Initialize Gemma 3N specific caches
    await caches.open(CACHE_NAMES.GEMMA_VOICE);
    await caches.open(CACHE_NAMES.GEMMA_MULTIMODAL);
    await caches.open(CACHE_NAMES.GEMMA_MODELS);
    console.log('🧠 Gemma 3N specialized caches initialized.');

    // Pre-cache essential API data
    const dataCache = await caches.open(CACHE_NAMES.DATA_API);
    await Promise.allSettled(
      API_ENDPOINTS_TO_HANDLE
        .filter(url => !url.includes('/api/submit') && !url.includes('/api/optimize')) // Only GET endpoints initially
        .map(url =>
          fetch(url).then(response => {
            if (response.ok) return dataCache.put(url, response.clone());
            throw new Error(`Failed to pre-cache data for ${url}: ${response.status}`);
          }).catch(err => console.warn(`⚠️ Failed to pre-cache ${url}:`, err))
        )
    );
    console.log('🔄 Essential API data pre-cached.');
  },

  async cleanupOldCaches() {
    const cacheNames = await caches.keys();
    const currentCaches = Object.values(CACHE_NAMES);
    const oldCaches = cacheNames.filter(name => !currentCaches.includes(name));
    
    console.log(`🧹 Cleaning up ${oldCaches.length} old caches`);
    return Promise.all(oldCaches.map(cacheName => caches.delete(cacheName)));
  },

  async limitCacheSize(cacheName, maxItemsOrBytes) {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    
    // Check if limit is in bytes (for Gemma 3N caches) or items
    if (typeof maxItemsOrBytes === 'number' && maxItemsOrBytes > 1000) { // Assuming bytes if over 1KB
        let currentSize = 0;
        const entries = await Promise.all(keys.map(async key => {
            const response = await cache.match(key);
            const size = parseInt(response.headers.get('content-length') || '0');
            currentSize += size;
            return { key, size };
        }));

        // Sort by oldest first (assuming keys are added chronologically or can be sorted by a 'date' header)
        // A more robust solution would store timestamps in IndexedDB alongside the cache key.
        // For simplicity, we'll just sort by default key order here.
        entries.sort((a, b) => a.key.url.localeCompare(b.key.url)); // Alphabetical for demo

        while (currentSize > maxItemsOrBytes && entries.length > 0) {
            const oldestEntry = entries.shift();
            await cache.delete(oldestEntry.key);
            currentSize -= oldestEntry.size;
            console.log(`🗑️ Deleted oldest entry from ${cacheName} to limit size: ${oldestEntry.key.url}`);
        }

    } else if (keys.length > maxItemsOrBytes) { // Limit by number of items
      const toDelete = keys.slice(0, keys.length - maxItemsOrBytes); // Delete the oldest items
      await Promise.all(toDelete.map(key => cache.delete(key)));
      console.log(`🗑️ Limited cache ${cacheName} to ${maxItemsOrBytes} items, deleted ${toDelete.length}`);
    }
  },

  async performMaintenanceCleanup() {
    console.log("🧹 Performing periodic cache maintenance...");
    
    await Promise.all([
      this.limitCacheSize(CACHE_NAMES.DYNAMIC_ASSETS, CACHE_LIMITS.DYNAMIC),
      this.limitCacheSize(CACHE_NAMES.DATA_API, CACHE_LIMITS.API),
      this.limitCacheSize(CACHE_NAMES.IMAGES, CACHE_LIMITS.IMAGES),
      this.limitCacheSize(CACHE_NAMES.PAGES, CACHE_LIMITS.PAGES),
      this.limitCacheSize(CACHE_NAMES.GEMMA_VOICE, CACHE_LIMITS.GEMMA_VOICE),
      this.limitCacheSize(CACHE_NAMES.GEMMA_MULTIMODAL, CACHE_LIMITS.GEMMA_MULTIMODAL),
      this.limitCacheSize(CACHE_NAMES.GEMMA_MODELS, CACHE_LIMITS.GEMMA_MODELS),
      databaseManager.cleanupOldEntries()
    ]);
    
    console.log("✅ Cache maintenance completed.");
  }
};

// ================================================================================
// 🗄️ DATABASE MANAGER
// ================================================================================

const databaseManager = {
  async initialize() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(SW_CONFIG.DB_NAME, SW_CONFIG.DB_VERSION);
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        console.log(`🗄️ Upgrading database schema to version ${SW_CONFIG.DB_VERSION}...`);
        
        // Create/Update all necessary object stores
        Object.values(DB_STORES).forEach(storeName => {
          if (!db.objectStoreNames.contains(storeName)) {
            const store = db.createObjectStore(storeName, { 
              keyPath: "id", 
              autoIncrement: true 
            });
            
            // Add common indexes for queue/log stores
            if (storeName.includes('Queue') || storeName.includes('Log')) {
              store.createIndex("timestamp", "timestamp", { unique: false });
              store.createIndex("status", "status", { unique: false });
              store.createIndex("priority", "priority", { unique: false });
              store.createIndex("endpoint", "endpoint", { unique: false });
            }
          }
        });
      };
      
      request.onsuccess = () => {
        console.log("✅ Database initialized/opened successfully.");
        resolve(request.result);
      };
      
      request.onerror = () => {
        console.error("❌ Database initialization failed:", request.error);
        reject(request.error);
      };
    });
  },

  async updateSchema() {
    // The `initialize` function already handles schema upgrades via `onupgradeneeded`.
    // We just need to call it to ensure the latest schema is applied.
    try {
      await this.initialize();
      console.log("✅ Database schema verified/updated.");
    } catch (error) {
      console.error("❌ Failed to verify/update database schema:", error);
      throw error;
    }
  },

  async cleanupOldEntries() {
    try {
      const db = await this.initialize();
      const cutoffTime = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days old
      
      // Clean up old analytics logs
      const analyticsTx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
      const analyticsStore = analyticsTx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
      const analyticsIndex = analyticsStore.index("timestamp");
      
      const oldAnalyticsCursor = await analyticsIndex.openCursor(IDBKeyRange.upperBound(cutoffTime));
      let deletedAnalyticsCount = 0;
      while (oldAnalyticsCursor) {
        await analyticsStore.delete(oldAnalyticsCursor.primaryKey);
        deletedAnalyticsCount++;
        oldAnalyticsCursor = await oldAnalyticsCursor.continue();
      }
      console.log(`🗄️ Cleaned up ${deletedAnalyticsCount} old analytics entries.`);

      // Clean up old device status logs (e.g., keep only last 30 days)
      const deviceStatusCutoff = Date.now() - (30 * 24 * 60 * 60 * 1000); // 30 days
      const deviceStatusTx = db.transaction([DB_STORES.DEVICE_STATUS_LOG], "readwrite");
      const deviceStatusStore = deviceStatusTx.objectStore(DB_STORES.DEVICE_STATUS_LOG);
      const deviceStatusIndex = deviceStatusStore.index("timestamp");

      const oldDeviceStatusCursor = await deviceStatusIndex.openCursor(IDBKeyRange.upperBound(deviceStatusCutoff));
      let deletedDeviceStatusCount = 0;
      while (oldDeviceStatusCursor) {
        await deviceStatusStore.delete(oldDeviceStatusCursor.primaryKey);
        deletedDeviceStatusCount++;
        oldDeviceStatusCursor = await oldDeviceStatusCursor.continue();
      }
      console.log(`🗄️ Cleaned up ${deletedDeviceStatusCount} old device status logs.`);

    } catch (error) {
      console.error("❌ Failed to cleanup old database entries:", error);
    }
  },

  async getItemsFromStore(storeName, query = null) {
    const db = await this.initialize();
    const tx = db.transaction([storeName], "readonly");
    const store = tx.objectStore(storeName);
    if (query) {
      // You can add more complex querying here using indexes if needed
      return await store.get(query);
    }
    return await store.getAll();
  },

  async addItemToStore(storeName, item) {
    const db = await this.initialize();
    const tx = db.transaction([storeName], "readwrite");
    const store = tx.objectStore(storeName);
    return await store.add(item);
  },

  async updateItemInStore(storeName, item) {
    const db = await this.initialize();
    const tx = db.transaction([storeName], "readwrite");
    const store = tx.objectStore(storeName);
    return await store.put(item);
  },

  async deleteItemFromStore(storeName, id) {
    const db = await this.initialize();
    const tx = db.transaction([storeName], "readwrite");
    const store = tx.objectStore(storeName);
    return await store.delete(id);
  },

  async clearStore(storeName) {
    const db = await this.initialize();
    const tx = db.transaction([storeName], "readwrite");
    const store = tx.objectStore(storeName);
    return await store.clear();
  }
};

// ================================================================================
// 🔄 SYNC MANAGER
// ================================================================================

const syncManager = {
  getSyncHandler(tag) {
    const syncHandlers = {
      [SYNC_TAGS.BROADCAST]: () => this.syncQueuedItems(DB_STORES.BROADCAST_QUEUE, "/broadcast"),
      [SYNC_TAGS.CROWD_REPORT]: () => this.syncQueuedItems(DB_STORES.CROWD_REPORT_QUEUE, "/api/submit-crowd-report", true),
      [SYNC_TAGS.TRIAGE]: () => this.syncQueuedItems(DB_STORES.TRIAGE_QUEUE, "/submit-triage", true),
      [SYNC_TAGS.EMERGENCY_REPORT]: () => this.syncQueuedItems(DB_STORES.EMERGENCY_REPORT_QUEUE, "/api/submit-emergency-report", true),
      [SYNC_TAGS.WEATHER_RISK]: () => this.syncQueuedItems(DB_STORES.WEATHER_RISK_QUEUE, "/predict-risk"),
      [SYNC_TAGS.IMAGE_ANALYSIS]: () => this.syncQueuedItems(DB_STORES.IMAGE_ANALYSIS_QUEUE, "/analyze-image", true),
      [SYNC_TAGS.FEEDBACK]: () => this.syncQueuedItems(DB_STORES.FEEDBACK_QUEUE, "/api/feedback", true),
      [SYNC_TAGS.DEVICE_STATUS]: () => this.syncQueuedItems(DB_STORES.DEVICE_STATUS_LOG, "/api/device-status"), // Assuming device status logs are sent periodically
      [SYNC_TAGS.EXPORT_REQUESTS]: () => this.syncQueuedItems(DB_STORES.EXPORT_REQUEST_QUEUE, "/api/export-reports", true),
      [SYNC_TAGS.GEMMA_VOICE]: () => this.syncQueuedItems(DB_STORES.GEMMA_VOICE_SYNC_QUEUE, "/api/submit-voice-emergency-report", true),
      [SYNC_TAGS.GEMMA_MULTIMODAL]: () => this.syncQueuedItems(DB_STORES.GEMMA_MULTIMODAL_SYNC_QUEUE, "/api/submit-damage-assessment", true),
    };
    
    return syncHandlers[tag];
  },

  async queueForSync(request, syncTag, storeName, options = {}) {
    try {
      let data;
      const contentType = request.headers.get("content-type");
      let isFormData = false;

      if (contentType && contentType.includes("application/json")) {
        data = await request.json();
      } else if (contentType && contentType.includes("multipart/form-data")) {
        // For multipart, we need to reconstruct the FormData object on sync
        const formData = await request.formData();
        data = {};
        for (let [key, value] of formData.entries()) {
            // Handle files specifically, store as Blob or dataURL if small
            if (value instanceof File || value instanceof Blob) {
                const reader = new FileReader();
                reader.readAsDataURL(value);
                await new Promise(resolve => {
                    reader.onloadend = () => {
                        data[key] = {
                            __isBlob: true,
                            data: reader.result,
                            type: value.type,
                            name: value.name || 'blob'
                        };
                        resolve();
                    };
                });
            } else {
                data[key] = value;
            }
        }
        isFormData = true;
      } else {
        data = await request.text();
      }
      
      const queueItem = {
        id: options.id || Date.now(), // Allow custom ID for specific needs
        data,
        url: request.url,
        method: request.method,
        headers: Object.fromEntries(request.headers.entries()),
        timestamp: Date.now(),
        priority: options.priority || "normal",
        type: options.type || "unknown",
        endpoint: options.endpoint || request.url,
        retryCount: 0,
        maxRetries: options.maxRetries || 3,
        status: "pending",
        isFormData // Indicate if original request was FormData
      };
      
      await databaseManager.addItemToStore(storeName, queueItem);
      
      console.log(`📝 Queued item for sync: ${syncTag} (ID: ${queueItem.id})`);
      
      if ("sync" in self.registration) {
        await self.registration.sync.register(syncTag);
      }
      
      return queueItem; // Return the queued item for potential client-side use
    } catch (error) {
      console.error("❌ Failed to queue for sync:", error);
      throw error; // Re-throw to indicate failure
    }
  },

  async syncQueuedItems(storeName, endpoint, isFormDataFlag = false) {
    try {
      const items = await databaseManager.getItemsFromStore(storeName);
      
      console.log(`🔄 Syncing ${items.length} items from ${storeName}. Endpoint: ${endpoint}`);
      
      for (const item of items) {
        try {
          let body;
          let headers = { ...item.headers };
          
          if (item.isFormData) { // Use the stored isFormData flag
            body = new FormData();
            for (const key in item.data) {
                if (item.data[key] && item.data[key].__isBlob) {
                    const blobData = item.data[key];
                    const blob = await (await fetch(blobData.data)).blob(); // Convert DataURL back to Blob
                    body.append(key, blob, blobData.name);
                } else {
                    body.append(key, item.data[key]);
                }
            }
            delete headers["content-type"]; // FormData sets its own Content-Type with boundary
          } else {
            body = JSON.stringify(item.data);
            headers["content-type"] = "application/json";
          }
          
          const response = await utils.fetchWithTimeout(new Request(item.url, { // Use item.url as some endpoints might be dynamic
            method: item.method,
            headers,
            body
          }));
          
          if (response.ok) {
            await databaseManager.deleteItemFromStore(storeName, item.id);
            console.log(`✅ Successfully synced item ${item.id} from ${storeName}.`);
            // Notify clients about successful sync (e.g., update UI)
            messageManager.notifyClients({ type: 'ITEM_SYNCED', payload: { id: item.id, store: storeName } });
          } else {
            await this.handleSyncFailure(storeName, item, response.status);
          }
        } catch (error) {
          console.error(`❌ Error syncing item ${item.id} from ${storeName}:`, error);
          await this.handleSyncFailure(storeName, item, error.message || 'Network Error');
        }
      }
      
    } catch (error) {
      console.error(`❌ Failed to sync all items from ${storeName}:`, error);
    }
  },

  async handleSyncFailure(storeName, item, errorDetails) {
    item.retryCount = (item.retryCount || 0) + 1;
    item.status = "failed"; // Set status to failed initially
    item.lastError = errorDetails;
    
    if (item.retryCount >= item.maxRetries) {
      console.error(`❌ Max retries (${item.maxRetries}) exceeded for item ${item.id} in ${storeName}. Removing from queue.`);
      item.status = "permanently_failed";
      await databaseManager.deleteItemFromStore(storeName, item.id); // Or move to a 'failed' store for inspection
      messageManager.notifyClients({ type: 'ITEM_SYNC_FAILED_PERMANENTLY', payload: { id: item.id, store: storeName, error: errorDetails } });
    } else {
      console.warn(`⚠️ Retrying item ${item.id} from ${storeName}. Retry count: ${item.retryCount}.`);
      item.status = "retrying";
      await databaseManager.updateItemInStore(storeName, item);
      messageManager.notifyClients({ type: 'ITEM_SYNC_RETRYING', payload: { id: item.id, store: storeName, retryCount: item.retryCount } });
    }
    // Log sync attempt/failure for analytics
    analyticsManager.logEvent('sync_attempt_result', {
      id: item.id,
      store: storeName,
      success: false,
      retryCount: item.retryCount,
      error: errorDetails
    });
  },

  async checkPendingSyncs() {
    try {
      const queueStores = Object.values(DB_STORES).filter(store => store.includes('Queue'));
      
      for (const storeName of queueStores) {
        const items = await databaseManager.getItemsFromStore(storeName);
        if (items.length > 0) {
          console.log(`📝 Found ${items.length} pending items in ${storeName}. Attempting to register sync.`);
          // Find the corresponding sync tag
          const syncTagKey = Object.keys(SYNC_TAGS).find(key => 
            DB_STORES[key.replace('-', '_').toUpperCase() + '_QUEUE'] === storeName
          );
          if (syncTagKey && "sync" in self.registration) {
            await self.registration.sync.register(SYNC_TAGS[syncTagKey]);
            console.log(`✅ Registered background sync for ${SYNC_TAGS[syncTagKey]}`);
          }
        }
      }
    } catch (error) {
      console.error("❌ Failed to check pending syncs:", error);
    }
  },

  async forceSyncAllQueues() {
    console.log("🔄 Force syncing all queues by registering all sync tags...");
    
    const syncPromises = Object.values(SYNC_TAGS).map(async (tag) => {
      if ("sync" in self.registration) {
        try {
          await self.registration.sync.register(tag);
          console.log(`✅ Registered force sync for tag: ${tag}`);
        } catch (error) {
          console.warn(`⚠️ Failed to register force sync for ${tag}:`, error);
        }
      }
    });
    
    await Promise.all(syncPromises);
    console.log("✅ All known sync queues registered for immediate force sync.");
  }
};

// ================================================================================
// 🌐 REQUEST HANDLER
// ================================================================================

const requestHandler = {
  // Strategy: Network-only for modifications, then queue if offline
  async handleDataModificationRequest(request) {
    const url = new URL(request.url);
    const syncQueueMap = {
      '/api/submit-crowd-report': SYNC_TAGS.CROWD_REPORT,
      '/submit-triage': SYNC_TAGS.TRIAGE,
      '/api/submit-emergency-report': SYNC_TAGS.EMERGENCY_REPORT,
      '/broadcast': SYNC_TAGS.BROADCAST,
      '/predict-risk': SYNC_TAGS.WEATHER_RISK,
      '/analyze-image': SYNC_TAGS.IMAGE_ANALYSIS,
      '/api/feedback': SYNC_TAGS.FEEDBACK,
      '/api/submit-voice-emergency-report': SYNC_TAGS.GEMMA_VOICE,
      '/api/submit-damage-assessment': SYNC_TAGS.GEMMA_MULTIMODAL,
      '/api/context-analysis': SYNC_TAGS.EMERGENCY_REPORT, // Treat high-value AI analysis as critical report
      '/api/optimize-ai-settings': SYNC_TAGS.ADMIN_ACTIONS,
      '/api/device-status': SYNC_TAGS.DEVICE_STATUS,
      '/api/export-reports': SYNC_TAGS.EXPORT_REQUESTS,
      '/api/patients': SYNC_TAGS.TRIAGE, // For updates to patients
      '/patients/': SYNC_TAGS.TRIAGE // For updates/discharges
    };

    const endpointKey = Object.keys(syncQueueMap).find(key => url.pathname.startsWith(key));
    const syncTag = endpointKey ? syncQueueMap[endpointKey] : SYNC_TAGS.CROWD_REPORT; // Default fallback

    let storeName;
    switch (syncTag) {
      case SYNC_TAGS.BROADCAST: storeName = DB_STORES.BROADCAST_QUEUE; break;
      case SYNC_TAGS.CROWD_REPORT: storeName = DB_STORES.CROWD_REPORT_QUEUE; break;
      case SYNC_TAGS.TRIAGE: storeName = DB_STORES.TRIAGE_QUEUE; break;
      case SYNC_TAGS.EMERGENCY_REPORT: storeName = DB_STORES.EMERGENCY_REPORT_QUEUE; break;
      case SYNC_TAGS.WEATHER_RISK: storeName = DB_STORES.WEATHER_RISK_QUEUE; break;
      case SYNC_TAGS.IMAGE_ANALYSIS: storeName = DB_STORES.IMAGE_ANALYSIS_QUEUE; break;
      case SYNC_TAGS.FEEDBACK: storeName = DB_STORES.FEEDBACK_QUEUE; break;
      case SYNC_TAGS.DEVICE_STATUS: storeName = DB_STORES.DEVICE_STATUS_LOG; break; // Logs should also be queued
      case SYNC_TAGS.EXPORT_REQUESTS: storeName = DB_STORES.EXPORT_REQUEST_QUEUE; break;
      case SYNC_TAGS.GEMMA_VOICE: storeName = DB_STORES.GEMMA_VOICE_SYNC_QUEUE; break;
      case SYNC_TAGS.GEMMA_MULTIMODAL: storeName = DB_STORES.GEMMA_MULTIMODAL_SYNC_QUEUE; break;
      case SYNC_TAGS.ADMIN_ACTIONS: storeName = DB_STORES.ADMIN_CACHE; break; // Store admin actions if not immediately synced
      default: storeName = DB_STORES.CROWD_REPORT_QUEUE;
    }

    try {
      const networkResponse = await utils.fetchWithTimeout(request.clone());
      if (networkResponse.ok) {
        console.log(`⬆️ Successfully sent ${request.method} request to ${url.pathname}`);
        return networkResponse;
      }
      // If network response is not OK, fall through to error/queue handling
      throw new Error(`Network response not OK: ${networkResponse.status}`);
    } catch (error) {
      console.warn(`📡 ${request.method} request to ${url.pathname} failed: ${error.message}. Queuing...`);
      try {
        const queuedItem = await syncManager.queueForSync(request.clone(), syncTag, storeName, {
          priority: (syncTag === SYNC_TAGS.BROADCAST || syncTag === SYNC_TAGS.EMERGENCY_REPORT || syncTag === SYNC_TAGS.TRIAGE) ? "critical" : "normal",
          endpoint: url.pathname
        });
        // Return a 202 Accepted status for queued requests
        return new Response(JSON.stringify({
          status: "queued_for_sync",
          offline: true,
          id: queuedItem.id,
          message: `Request queued for background sync. Priority: ${queuedItem.priority}.`
        }), {
          status: 202,
          headers: { "Content-Type": "application/json" }
        });
      } catch (queueError) {
        console.error(`❌ Failed to queue request for ${url.pathname}:`, queueError);
        return new Response(JSON.stringify({
          status: "offline_failed",
          error: "Failed to queue request for sync. No network available.",
          message: queueError.message
        }), {
          status: 503,
          headers: { "Content-Type": "application/json" }
        });
      }
    }
  },

  // Strategy: Cache-first for critical static assets, then network, then fallback
  async handleStaticAsset(request) {
    const url = new URL(request.url);
    const cacheName = url.pathname.includes('/images/') ? CACHE_NAMES.IMAGES : CACHE_NAMES.CORE_APP;
    const cacheLimit = url.pathname.includes('/images/') ? CACHE_LIMITS.IMAGES : CACHE_LIMITS.DYNAMIC;

    try {
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        // Update cache in background
        utils.fetchAndCache(request, cacheName, cacheLimit).catch(err => console.warn('Background cache update failed for static asset:', err));
        return cachedResponse;
      }
      // If not in cache, try network
      const networkResponse = await fetch(request);
      if (networkResponse.ok) {
        await utils.fetchAndCache(request, cacheName, cacheLimit);
        return networkResponse;
      }
      throw new Error(`Static asset network response not ok: ${networkResponse.status}`);
    } catch (error) {
      console.warn(`📄 Static asset ${url.pathname} failed: ${error.message}. Serving fallback if available.`);
      if (url.pathname.includes('.png') || url.pathname.includes('.jpg') || url.pathname.includes('.jpeg')) {
        return fallbackGenerator.generateOfflineImagePlaceholder();
      }
      if (url.pathname.includes('.css')) {
        return fallbackGenerator.generateOfflineCSSFallback();
      }
      // For other static assets, return a generic error response
      return new Response('Asset not available offline', { status: 404 });
    }
  },

  // Strategy: Network-first, then cache, then offline JSON fallback
  async handleApiRequest(request) {
    const url = new URL(request.url);
    const isGemmaApi = (url.pathname.includes('/api/ai-model-status') || url.pathname.includes('/api/device-performance'));
    const cacheName = isGemmaApi ? CACHE_NAMES.GEMMA_MODELS : CACHE_NAMES.DATA_API;
    const cacheLimit = isGemmaApi ? CACHE_LIMITS.GEMMA_MODELS : CACHE_LIMITS.API;

    try {
      // Always try network first for API requests to get fresh data
      const networkResponse = await utils.fetchWithTimeout(request.clone());
      if (networkResponse.ok) {
        // Cache successful responses for future offline use
        await utils.fetchAndCache(request.clone(), cacheName, cacheLimit);
        return networkResponse;
      }
      // If network response is not OK, try cache
      throw new Error(`API network response not ok: ${networkResponse.status}`);
    } catch (error) {
      console.warn(`📡 API ${url.pathname} failed: ${error.message}. Attempting to serve from cache.`);
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        const headers = new Headers(cachedResponse.headers);
        headers.set('X-Served-By', 'ServiceWorker-Cache');
        headers.set('X-Cache-Date', new Date().toISOString());
        console.log(`📦 Serving cached API response for ${url.pathname}.`);
        return new Response(cachedResponse.body, { status: cachedResponse.status, statusText: cachedResponse.statusText, headers });
      }
      // If no network and no cache, provide a tailored JSON offline fallback
      console.log(`❌ No cached API response for ${url.pathname}. Providing offline JSON.`);
      return fallbackGenerator.generateOfflineApiResponse(request);
    }
  },

  // Strategy: Network-first for navigation, then critical page cache, then generic offline page
  async handlePageNavigation(request) {
    try {
      const networkResponse = await utils.fetchWithTimeout(request.clone());
      if (networkResponse.ok) {
        // Update the cache for this page in the background
        utils.fetchAndCache(request.clone(), CACHE_NAMES.PAGES, CACHE_LIMITS.PAGES).catch(err => console.warn('Background cache update failed for page:', err));
        return networkResponse;
      }
      throw new Error(`Navigation network response not ok: ${networkResponse.status}`);
    } catch (error) {
      console.warn(`🌐 Navigation to ${request.url} failed: ${error.message}. Serving cached page or offline fallback.`);
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        console.log(`📦 Serving cached page for ${request.url}.`);
        return cachedResponse;
      }
      // Fallback to a custom offline page
      console.log(`📴 No cached page for ${request.url}. Serving general offline page.`);
      const offlineResponse = await caches.match(OFFLINE_URL);
      return offlineResponse || fallbackGenerator.generateOfflinePageFallback(request.url);
    }
  },

  // Strategy: Network-only by default, no offline support unless explicitly handled
  async handleDefaultNetworkOnly(request) {
    try {
      return await utils.fetchWithTimeout(request);
    } catch (error) {
      console.warn(`⚠️ Network-only request to ${request.url} failed: ${error.message}.`);
      return new Response(JSON.stringify({
        error: "Service unavailable offline",
        message: `This operation requires an internet connection and cannot be queued or cached. (${error.message})`
      }), {
        status: 503,
        headers: { "Content-Type": "application/json" }
      });
    }
  },

  // Strategy: Network-first for generic GET, then dynamic cache
  async handleGenericGet(request) {
    const cacheName = CACHE_NAMES.DYNAMIC_ASSETS;
    const cacheLimit = CACHE_LIMITS.DYNAMIC;

    try {
      const networkResponse = await utils.fetchWithTimeout(request.clone());
      if (networkResponse.ok) {
        await utils.fetchAndCache(request.clone(), cacheName, cacheLimit);
        return networkResponse;
      }
      throw new Error(`Generic GET network response not ok: ${networkResponse.status}`);
    } catch (error) {
      console.warn(`🌍 Generic GET request to ${request.url} failed: ${error.message}. Checking dynamic cache.`);
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        console.log(`📦 Serving from dynamic cache for ${request.url}.`);
        return cachedResponse;
      }
      // If neither network nor cache, return a generic offline response
      return new Response('Content not available offline.', { status: 503 });
    }
  }
};

// ================================================================================
// 🎨 FALLBACK GENERATOR
// ================================================================================

const fallbackGenerator = {
  generateOfflineImagePlaceholder() {
    const svg = `
      <svg width="200" height="150" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f3f4f6"/>
        <text x="50%" y="50%" text-anchor="middle" dy="0.3em" fill="#6b7280" font-size="16">
          📷 Offline Image
        </text>
        <text x="50%" y="70%" text-anchor="middle" dy="0.3em" fill="#6b7280" font-size="12">
          (No network)
        </text>
      </svg>
    `;
    return new Response(svg, {
      headers: {
        'Content-Type': 'image/svg+xml',
        'Cache-Control': 'public, max-age=86400'
      }
    });
  },

  generateOfflineCSSFallback() {
    const css = `
      /* Minimal fallback CSS for offline */
      body { font-family: sans-serif; margin: 20px; background-color: #f8f8f8; color: #333; }
      .offline-notice { background-color: #fff3cd; border: 1px solid #ffeeba; color: #856404; padding: 10px; margin-bottom: 20px; border-radius: 5px; }
      h1, h2 { color: #555; }
      a { color: #007bff; text-decoration: none; }
      a:hover { text-decoration: underline; }
    `;
    return new Response(css, {
      headers: {
        'Content-Type': 'text/css',
        'Cache-Control': 'public, max-age=86400'
      }
    });
  },

  generateOfflinePageFallback(requestedUrl) {
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Offline - Emergency Response</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body { 
            font-family: Arial, sans-serif; 
            margin: 0; padding: 0;
            text-align: center; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
          }
          .container { 
            max-width: 600px; 
            margin: 0 auto; 
            padding: 2rem; 
            background: rgba(255,255,255,0.15);
            border-radius: 16px;
            backdrop-filter: blur(5px);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
          }
          .offline-icon { font-size: 4rem; margin-bottom: 1rem; }
          h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
          .message { 
            font-size: 1.1rem;
            background: rgba(254, 243, 199, 0.2); 
            padding: 1rem; 
            border-radius: 8px; 
            margin: 1rem 0; 
            border: 1px solid rgba(255,255,255,0.3);
          }
          .actions { margin-top: 2rem; }
          .btn { 
            display: inline-block; 
            padding: 0.75rem 1.5rem; 
            background: rgba(59, 130, 246, 0.8); 
            color: white; 
            text-decoration: none; 
            border-radius: 8px; 
            margin: 0.5rem; 
            border: 1px solid rgba(255,255,255,0.3);
            transition: all 0.3s ease;
          }
          .btn:hover {
            background: rgba(59, 130, 246, 1);
            transform: translateY(-2px);
          }
          .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #ef4444; /* Red for offline */
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse-red 2s infinite;
          }
          @keyframes pulse-red {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            50% { opacity: 0.8; box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="offline-icon">😔</div>
          <h1><span class="status-indicator" id="network-status-indicator"></span>You're Offline</h1>
          <div class="message">
            <p>The page <strong>"${requestedUrl}"</strong> isn't fully available offline. But don't worry!</p>
            <p>Your emergency reports and key features still work.</p>
            <p><strong>Reports you submit will be saved and synced automatically when you're back online.</strong></p>
          </div>
          <div class="actions">
            <a href="/" class="btn">🏠 Home</a>
            <a href="/submit-report" class="btn">🚨 New Report</a>
            <a href="/crowd-reports" class="btn">📊 View Offline Reports</a>
            <a href="/sync-status" class="btn">🔄 Check Sync Status</a>
          </div>
          <p style="font-size: 0.9em; margin-top: 2rem;">Last updated: ${new Date().toLocaleTimeString()}</p>
          <script>
            // Simple client-side network status update
            function updateNetworkStatus() {
              const indicator = document.getElementById('network-status-indicator');
              if (navigator.onLine) {
                indicator.style.backgroundColor = '#10b981'; // Green for online
                indicator.style.animation = 'none';
                // Optional: redirect after a short delay when online
                setTimeout(() => window.location.reload(), 1500);
              } else {
                indicator.style.backgroundColor = '#ef4444'; // Red for offline
                indicator.style.animation = 'pulse-red 2s infinite';
              }
            }
            window.addEventListener('online', updateNetworkStatus);
            window.addEventListener('offline', updateNetworkStatus);
            updateNetworkStatus(); // Initial check
          </script>
        </div>
      </body>
      </html>
    `;
    return new Response(html, {
      headers: {
        'Content-Type': 'text/html',
        'Cache-Control': 'no-cache'
      }
    });
  },

  generateOfflineApiResponse(request) {
    const url = new URL(request.url);
    const commonError = {
      success: false,
      error: "Network Unavailable",
      offline: true,
      timestamp: new Date().toISOString()
    };

    if (url.pathname.includes('/api/ai-model-status') || url.pathname.includes('/api/device-performance')) {
      return new Response(JSON.stringify({
        ...commonError,
        message: "AI model status or device performance data is not available offline. Displaying cached data if available, otherwise no data."
      }), { status: 503, headers: { 'Content-Type': 'application/json', 'X-Offline-Response': 'true' } });
    }

    // Generic API offline response
    return new Response(JSON.stringify({
      ...commonError,
      message: `API endpoint ${url.pathname} is not accessible offline.`
    }), { status: 503, headers: { 'Content-Type': 'application/json', 'X-Offline-Response': 'true' } });
  }
};

// ================================================================================
// 📱 NOTIFICATION MANAGER
// ================================================================================

const notificationManager = {
  handlePushNotification(event) {
    console.log("📱 Push notification received");
    let notificationData = {
      title: "Emergency Alert",
      body: "Emergency alert received",
      icon: "/static/icons/icon-192x192.png",
      badge: "/static/icons/badge-72x72.png",
      vibrate: [200, 100, 200, 100, 200],
      requireInteraction: false,
      tag: "emergency-alert",
      data: {}
    };
    
    if (event.data) {
      try {
        const data = event.data.json();
        notificationData = {
          ...notificationData, // Keep defaults
          title: data.title || notificationData.title,
          body: data.body || data.message || notificationData.body,
          icon: data.icon || notificationData.icon,
          badge: data.badge || notificationData.badge,
          data: data, // Store full data for click handling
          actions: [
            { action: "view", title: "View Details", icon: "/static/icons/view.png" }, // Assuming you have these icons
            { action: "dismiss", title: "Dismiss", icon: "/static/icons/dismiss.png" }
          ],
          vibrate: data.urgent ? [300, 100, 300, 100, 300, 100, 300] : notificationData.vibrate,
          requireInteraction: data.urgent || false, // Persistent notification for urgent
          tag: data.tag || notificationData.tag,
          silent: data.silent || false
        };
        
      } catch (error) {
        console.error("❌ Failed to parse push notification data:", error);
      }
    }
    
    event.waitUntil(
      Promise.all([
        self.registration.showNotification(notificationData.title, notificationData),
        this.logNotificationEvent('received', notificationData)
      ])
    );
  },

  handleNotificationClick(event) {
    console.log("📱 Notification clicked:", event.action);
    event.notification.close(); // Close the notification once clicked

    const notificationData = event.notification.data || {};
    const targetUrl = notificationData.url || notificationData.link || "/";
    
    // Log the click action
    this.logNotificationEvent('clicked', notificationData, event.action);

    if (event.action === "view") {
      event.waitUntil(
        clients.matchAll({ type: 'window' }).then(clientsArr => {
          // Check if there's already an open window for the target URL
          for (const client of clientsArr) {
            if (client.url === targetUrl && 'focus' in client) {
              return client.focus();
            }
          }
          // Otherwise, open a new window
          if (clients.openWindow) {
            return clients.openWindow(targetUrl);
          }
        })
      );
    } else if (event.action === "dismiss") {
      // Handled by close() above and logged by general click handler
    } else {
      // Default action if no specific action button was clicked
      event.waitUntil(
        clients.matchAll({ type: 'window' }).then(clientsArr => {
          if (clientsArr.length > 0) {
            // Focus first existing client
            return clientsArr[0].focus();
          }
          // If no clients open, open a new one to the home page
          if (clients.openWindow) {
            return clients.openWindow("/");
          }
        })
      );
    }
  },

  async logNotificationEvent(eventType, notificationData, actionTaken = null) {
    try {
      await analyticsManager.logEvent(`notification_${eventType}`, {
        title: notificationData.title,
        body: notificationData.body,
        tag: notificationData.tag,
        urgent: notificationData.requireInteraction,
        action: actionTaken,
        url: notificationData.data.url || notificationData.data.link
      });
    } catch (error) {
      console.error(`❌ Failed to log notification ${eventType} event:`, error);
    }
  }
};

// ================================================================================
// 💬 MESSAGE MANAGER (Client-SW Communication)
// ================================================================================

const messageManager = {
  handleMessage(event) {
    const { type, payload } = event.data || {};
    
    switch (type) {
      case "CLEANUP_CACHES":
        event.waitUntil(cacheManager.performMaintenanceCleanup());
        break;
        
      case "FORCE_SYNC":
        event.waitUntil(syncManager.forceSyncAllQueues());
        break;
        
      case "GET_OFFLINE_STATUS":
        event.waitUntil(this.sendOfflineStatus(event.source));
        break;
        
      case "CLEAR_SYNC_QUEUE":
        event.waitUntil(databaseManager.clearStore(payload.storeName));
        break;
        
      case "GET_SYNC_STATUS":
        event.waitUntil(this.sendSyncStatus(event.source));
        break;

      case "LOG_ANALYTICS_EVENT":
        event.waitUntil(analyticsManager.logEvent(payload.eventType, payload.eventData));
        break;

      case "CACHE_GEMMA3N_DATA":
        // This is a direct request to cache specific Gemma 3N data, e.g., small model updates
        // Payload should contain { url: ..., data: ... }
        event.waitUntil(utils.cacheCustomData(payload.url, payload.data, CACHE_NAMES.GEMMA_MODELS));
        break;
        
      default:
        console.warn(`⚠️ Unknown message type received: ${type}`, payload);
    }
  },

  async sendOfflineStatus(source) {
    try {
      const queueCounts = {};
      const totalPending = (await syncManager.checkPendingSyncs()).totalPending; // Use the function to get fresh counts
      
      for (const storeName of Object.values(DB_STORES)) {
        if (storeName.includes('Queue')) {
          queueCounts[storeName] = await databaseManager.getItemsFromStore(storeName).then(items => items.length);
        }
      }
      
      source.postMessage({
        type: 'OFFLINE_STATUS_RESPONSE',
        payload: {
          version: SW_CONFIG.VERSION,
          queueCounts,
          totalPendingSyncs: totalPending,
          timestamp: Date.now(),
          cacheInfo: await this.getDetailedCacheInfo(),
          online: navigator.onLine
        }
      });
    } catch (error) {
      console.error("❌ Failed to get offline status:", error);
    }
  },

  async sendSyncStatus(source) {
    try {
      const syncStatus = {};
      const queueStores = Object.values(DB_STORES).filter(name => name.includes('Queue'));

      for (const storeName of queueStores) {
        const items = await databaseManager.getItemsFromStore(storeName);
        syncStatus[storeName] = {
          total: items.length,
          pending: items.filter(item => item.status === 'pending').length,
          retrying: items.filter(item => item.status === 'retrying').length,
          failed: items.filter(item => item.status === 'permanently_failed').length
        };
      }
      
      source.postMessage({
        type: 'SYNC_STATUS_RESPONSE',
        payload: {
          syncStatus,
          timestamp: Date.now(),
          online: navigator.onLine
        }
      });
    } catch (error) {
      console.error("❌ Failed to get sync status:", error);
    }
  },

  async getDetailedCacheInfo() {
    const cacheNames = await caches.keys();
    const cacheInfo = {};
    
    for (const name of cacheNames) {
      const cache = await caches.open(name);
      const keys = await cache.keys();
      let size = 0;
      for (const key of keys) {
        const response = await cache.match(key);
        size += parseInt(response.headers.get('content-length') || '0');
      }
      cacheInfo[name] = { itemCount: keys.length, totalBytes: size };
    }
    return cacheInfo;
  },

  notifyClientsOfOfflineReadiness() {
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage({
          type: 'SERVICE_WORKER_READY',
          payload: {
            version: SW_CONFIG.VERSION,
            timestamp: Date.now(),
            status: "active",
            features: {
              caching: true, sync: true, notifications: true, analytics: true,
              gemma3nIntegration: true
            }
          }
        });
      });
    });
  },

  notifyClients(message) {
    self.clients.matchAll().then(clients => {
      clients.forEach(client => client.postMessage(message));
    });
  }
};

// ================================================================================
// 📊 ANALYTICS MANAGER
// ================================================================================

const analyticsManager = {
  async initializeOfflineAnalytics() {
    try {
      await this.logEvent('sw_installed', { version: SW_CONFIG.VERSION });
      console.log("📊 Offline analytics initialized.");
    } catch (error) {
      console.error("❌ Failed to initialize offline analytics:", error);
    }
  },

  async logEvent(eventType, eventData = {}) {
    try {
      await databaseManager.addItemToStore(DB_STORES.OFFLINE_ANALYTICS, {
        event: eventType,
        timestamp: Date.now(),
        data: eventData,
        version: SW_CONFIG.VERSION
      });
    } catch (error) {
      console.error("❌ Failed to log analytics event:", error);
    }
  },

  async syncAnalytics() {
    try {
      const analyticsEvents = await databaseManager.getItemsFromStore(DB_STORES.OFFLINE_ANALYTICS);
      if (analyticsEvents.length === 0) return;

      const response = await utils.fetchWithTimeout(new Request('/api/log-analytics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ events: analyticsEvents })
      }));

      if (response.ok) {
        await databaseManager.clearStore(DB_STORES.OFFLINE_ANALYTICS);
        console.log(`✅ Successfully synced ${analyticsEvents.length} analytics events.`);
      } else {
        console.warn(`⚠️ Failed to sync analytics: ${response.status}`);
      }
    } catch (error) {
      console.error("❌ Error during analytics sync:", error);
    }
  }
};

// ================================================================================
// 🔧 UTILITIES
// ================================================================================

const utils = {
  async fetchWithTimeout(request, timeout = SW_CONFIG.NETWORK_TIMEOUT) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(request, {
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error; // Propagate the error for further handling
    }
  },

  async fetchAndCache(request, cacheName, cacheLimit) {
    try {
      const response = await fetch(request.clone());
      if (response.ok) {
        const cache = await caches.open(cacheName);
        await cache.put(request, response.clone());
        await cacheManager.limitCacheSize(cacheName, cacheLimit);
      }
      return response;
    } catch (error) {
      console.warn(`Background fetch and cache failed for ${request.url} in ${cacheName}:`, error);
      throw error; // Re-throw so original request can use cache if needed
    }
  },

  async cacheCustomData(url, data, cacheName) {
    try {
      const cache = await caches.open(cacheName);
      const response = new Response(JSON.stringify(data), {
        headers: {
          'Content-Type': 'application/json',
          'Date': new Date().toISOString() // Important for cache age limits
        }
      });
      await cache.put(url, response);
      await cacheManager.limitCacheSize(cacheName, CACHE_LIMITS.GEMMA_MODELS); // Apply limit
      console.log(`💾 Custom data cached to ${cacheName} for ${url}`);
    } catch (error) {
      console.error(`❌ Failed to cache custom data for ${url}:`, error);
    }
  }
};

// ================================================================================
// ❌ ERROR MANAGER
// ================================================================================

const errorManager = {
  async logServiceWorkerError(error) {
    console.error('❌ Service Worker internal error:', error);
    await analyticsManager.logEvent('sw_internal_error', {
      message: error.message || String(error),
      stack: error.stack,
      version: SW_CONFIG.VERSION
    });
  }
};

// ================================================================================
// 🌐 OFFLINE MANAGER (Periodic Tasks & Network Monitoring)
// ================================================================================

const offlineManager = {
  async setupCapabilities() {
    // Set up periodic cache cleanup (e.g., every hour)
    setInterval(() => cacheManager.performMaintenanceCleanup(), 1000 * 60 * 60);
    
    // Set up periodic analytics sync (e.g., every 5 minutes when online)
    setInterval(() => {
      if (navigator.onLine) {
        analyticsManager.syncAnalytics();
      }
    }, 1000 * 60 * 5);

    // Set up network status monitoring
    this.setupNetworkMonitoring();

    // Set up periodic performance monitoring for Gemma 3N optimization
    this.setupPerformanceMonitoring();
    
    console.log("🔧 Offline capabilities and periodic tasks configured.");
  },

  setupNetworkMonitoring() {
    let lastOnlineStatus = navigator.onLine;
    
    // Check network status periodically and react to changes
    setInterval(() => {
      const currentOnlineStatus = navigator.onLine;
      
      if (currentOnlineStatus !== lastOnlineStatus) {
        console.log(`🌐 Network status changed: ${currentOnlineStatus ? 'Online' : 'Offline'}`);
        
        if (currentOnlineStatus) {
          syncManager.forceSyncAllQueues(); // Attempt to sync all pending items immediately
          analyticsManager.logEvent('network_restored');
        } else {
          analyticsManager.logEvent('network_lost');
        }
        
        // Notify all open client windows about the network status change
        messageManager.notifyClients({
          type: 'NETWORK_STATUS_CHANGED',
          payload: {
            online: currentOnlineStatus,
            timestamp: Date.now()
          }
        });
        
        lastOnlineStatus = currentOnlineStatus;
      }
    }, 5000); // Check every 5 seconds
  },

  setupPerformanceMonitoring() {
    // Monitor device performance periodically for Gemma 3N optimization
    setInterval(async () => {
      try {
        const performance = await this.getDevicePerformanceMetrics();
        this.optimizeGemma3nBasedOnPerformance(performance);
        // Log performance periodically for analytics/debugging
        analyticsManager.logEvent('device_performance_snapshot', performance);
      } catch (error) {
        console.warn('⚠️ Error during performance monitoring:', error);
      }
    }, 30000); // Run every 30 seconds
  },

  async getDevicePerformanceMetrics() {
    // This is a simplified example. Real-world implementation might need WebAssembly
    // for more direct access or rely on companion native apps.
    // For browser context, we rely on navigator properties.
    const performanceData = {
      memory: navigator.deviceMemory || 4, // Estimated device memory in GB
      connection: navigator.connection ? navigator.connection.effectiveType : 'unknown',
      battery: null, // Placeholder
      timestamp: Date.now()
    };

    if ('getBattery' in navigator) {
      try {
        const battery = await navigator.getBattery();
        performanceData.battery = {
          level: Math.round(battery.level * 100),
          charging: battery.charging
        };
      } catch (error) {
        console.warn('Battery Status API not available or permission denied:', error);
      }
    }
    return performanceData;
  },

  async optimizeGemma3nBasedOnPerformance(performance) {
    const optimizations = [];
    
    // Low battery optimization
    if (performance.battery && performance.battery.level < 20 && !performance.battery.charging) {
      optimizations.push('battery_saving_mode');
      console.log('🔋 Low battery detected, recommending Gemma 3N power saving mode.');
      // Example: Tell client to switch to smaller Gemma model or reduce inference frequency
    }
    
    // Low memory optimization
    if (performance.memory < 2) { // Devices with < 2GB RAM
      optimizations.push('aggressive_memory_management');
      console.log('💾 Low device memory, recommending aggressive Gemma 3N memory optimization.');
      // Force immediate Gemma 3N cache cleanup if memory is critical
      await cacheManager.limitCacheSize(CACHE_NAMES.GEMMA_VOICE, CACHE_LIMITS.GEMMA_VOICE / 2);
      await cacheManager.limitCacheSize(CACHE_NAMES.GEMMA_MULTIMODAL, CACHE_LIMITS.GEMMA_MULTIMODAL / 2);
    }
    
    // Poor network optimization
    if (performance.connection === 'slow-2g' || performance.connection === '2g') {
      optimizations.push('network_constrained_mode');
      console.log('📶 Slow network detected, recommending Gemma 3N network optimization.');
      // Example: Tell client to reduce multimodal data upload quality/frequency
    }

    if (optimizations.length > 0) {
      messageManager.notifyClients({
        type: 'GEMMA3N_OPTIMIZATION_RECOMMENDATION',
        payload: {
          optimizations,
          currentPerformance: performance
        }
      });
    }
  }
};

// Log service worker initialized after all modules are defined
console.log(`🎯 Enhanced Service Worker ${SW_CONFIG.VERSION} loaded and ready.`);