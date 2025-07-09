// ============================================================================
// ENHANCED SERVICE WORKER - EMERGENCY RESPONSE SYSTEM
// Version: 2.2.0 - Organized & Optimized
// ============================================================================

// ============================================================================
// 📋 CONFIGURATION & CONSTANTS
// ============================================================================

const SW_CONFIG = {
  VERSION: "v2.2.0",
  DB_NAME: "EmergencyResponseDB",
  DB_VERSION: 4,
  NETWORK_TIMEOUT: 5000,
  CACHE_TIMEOUT: 1000
};

const CACHE_NAME = `disaster-assistant-cache-${SW_CONFIG.VERSION}`;
const OFFLINE_URL = "/offline.html";

// Cache Configuration
const CACHE_CONFIG = {
  STATIC: `${CACHE_NAME}-static`,
  DYNAMIC: `${CACHE_NAME}-dynamic`,
  API: `${CACHE_NAME}-api`,
  IMAGES: `${CACHE_NAME}-images`,
  PAGES: `${CACHE_NAME}-pages`,
  MAX_DYNAMIC_ITEMS: 50,
  MAX_API_ITEMS: 100,
  MAX_IMAGE_ITEMS: 30,
  MAX_PAGE_ITEMS: 20
};

// Sync Queue Configurations
const SYNC_QUEUES = {
  BROADCAST: "emergency-broadcast-sync",
  CROWD_REPORT: "crowd-report-sync",
  TRIAGE: "triage-assessment-sync",
  EMERGENCY_REPORT: "emergency-report-sync",
  WEATHER_RISK: "weather-risk-sync",
  IMAGE_ANALYSIS: "image-analysis-sync",
  FEEDBACK: "feedback-sync",
  DEVICE_STATUS: "device-status-sync",
  ADMIN_ACTIONS: "admin-actions-sync",
  EXPORT_REQUESTS: "export-requests-sync",
  USER_PREFERENCES: "user-preferences-sync"
};

// IndexedDB Store Configurations
const DB_STORES = {
  // Queue Stores
  BROADCAST_QUEUE: "broadcastQueue",
  REPORT_QUEUE: "crowdReportQueue",
  TRIAGE_QUEUE: "triageQueue",
  EMERGENCY_QUEUE: "emergencyReportQueue",
  WEATHER_QUEUE: "weatherRiskQueue",
  IMAGE_QUEUE: "imageAnalysisQueue",
  FEEDBACK_QUEUE: "feedbackQueue",
  EXPORT_QUEUE: "exportRequestQueue",
  
  // Data Stores
  SYNC_METADATA: "syncMetadata",
  OFFLINE_ANALYTICS: "offlineAnalytics",
  USER_PREFERENCES: "userPreferences",
  DEVICE_STATUS: "deviceStatusLog",
  SYNC_HISTORY: "syncHistoryLog",
  ARCHIVE_CACHE: "reportArchiveCache",
  ONBOARDING_STATE: "onboardingProgress",
  HELP_CACHE: "helpContentCache",
  ADMIN_CACHE: "adminDashboardCache"
};

// Critical Pages for Offline Access
const CRITICAL_PAGES = [
  "/", "/home", "/submit-report", "/submit-crowd-report", "/triage-form",
  "/triage-dashboard", "/patient-tracker", "/patient-list", "/crowd-reports",
  "/map-reports", "/view-reports", "/hazards", "/predict", "/live-generate",
  "/test-offline", "/offline.html", "/sync-status", "/device-status",
  "/report-archive", "/onboarding", "/admin-dashboard", "/feedback", "/help"
];

// Critical Assets for Offline Functionality
const CRITICAL_ASSETS = [
  "/", "/offline.html", "/static/css/styles.css", "/manifest.json",
  
  // Core JavaScript
  "/static/js/weather-risk.js", "/static/js/sync-queue.js", "/static/js/edge-ai.js",
  "/static/js/workers/broadcast.js", "/static/js/p2p/fallback-webrtc.js", "/static/js/idb.mjs",
  
  // Feature-specific JS
  "/static/js/triage-utils.js", "/static/js/pdf-export.js", "/static/js/patient-management.js",
  "/static/js/map-utils.js", "/static/js/sync-manager.js", "/static/js/device-monitor.js",
  "/static/js/archive-manager.js", "/static/js/onboarding-flow.js", "/static/js/admin-controls.js",
  "/static/js/feedback-handler.js", "/static/js/help-system.js",
  
  // External Dependencies
  "https://unpkg.com/leaflet/dist/leaflet.css",
  "https://unpkg.com/leaflet/dist/leaflet.js"
];

// API Endpoints for Offline Handling
const OFFLINE_API_ENDPOINTS = [
  "/api/submit-crowd-report", "/api/submit-emergency-report", "/submit-triage",
  "/predict-risk", "/analyze-sentiment", "/generate-report", "/broadcast",
  "/api/patients", "/api/triage-queue", "/api/hazard-analysis", "/api/sync-status",
  "/api/offline-reports", "/api/system-status", "/api/sync-queue", "/api/dashboard-stats",
  "/api/community-stats", "/api/recent-reports", "/api/feedback", "/api/help-search",
  "/api/device-diagnostics", "/api/export-reports", "/api/user-preferences", "/api/onboarding-progress"
];

// Page-specific Configuration
const PAGE_SPECIFIC_CONFIG = {
  "/admin-dashboard": { requiresAuth: true, fallbackData: "adminFallbackData", cacheStrategy: "cacheFirst" },
  "/device-status": { requiresRealTime: true, fallbackData: "deviceFallbackData", cacheStrategy: "networkFirst" },
  "/sync-status": { requiresRealTime: true, fallbackData: "syncFallbackData", cacheStrategy: "networkFirst" },
  "/report-archive": { requiresData: true, fallbackData: "archiveFallbackData", cacheStrategy: "cacheFirst" }
};

// Offline Fallback Data
const OFFLINE_FALLBACK_DATA = {
  adminFallbackData: {
    stats: { total_reports: 0, active_users: 1, avg_severity: 0 },
    message: "Offline mode - Limited admin functionality available"
  },
  deviceFallbackData: {
    status: "offline",
    message: "Device status unavailable while offline"
  },
  syncFallbackData: {
    queue: [],
    status: "offline", 
    message: "Sync status unavailable while offline"
  },
  archiveFallbackData: {
    reports: [],
    message: "Report archive unavailable while offline"
  }
};

// ============================================================================
// 🚀 SERVICE WORKER EVENT LISTENERS
// ============================================================================

// Installation Event
self.addEventListener("install", (event) => {
  console.log(`🔧 Service Worker ${SW_CONFIG.VERSION} installing...`);
  
  event.waitUntil(
    Promise.all([
      cacheManager.cacheStaticAssets(),
      cacheManager.cacheCriticalPages(),
      databaseManager.initialize(),
      analyticsManager.initializeOfflineAnalytics()
    ]).then(() => {
      console.log("✅ Service Worker installation complete");
    })
  );
  
  self.skipWaiting();
});

// Activation Event
self.addEventListener("activate", (event) => {
  console.log(`🚀 Service Worker ${SW_CONFIG.VERSION} activating...`);
  
  event.waitUntil(
    Promise.all([
      cacheManager.cleanupOldCaches(),
      self.clients.claim(),
      databaseManager.updateSchema(),
      offlineManager.setupCapabilities(),
      syncManager.checkPendingSyncs()
    ]).then(() => {
      console.log("✅ Service Worker activation complete");
      messageManager.notifyClientsOfOfflineReadiness();
    })
  );
});

// Fetch Event - Main Request Handler
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  const urlPath = url.pathname;
  
  if (!request.url.startsWith('http')) return;
  
  if (request.method === "POST") {
    requestHandler.handlePostRequest(event, request, urlPath);
  } else if (request.method === "GET") {
    requestHandler.handleGetRequest(event, request, urlPath);
  } else {
    event.respondWith(requestHandler.handleOtherMethods(request));
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

// Message Event
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

// Error Handling
self.addEventListener('error', (event) => {
  errorManager.logServiceWorkerError(event.error);
});

self.addEventListener('unhandledrejection', (event) => {
  errorManager.logServiceWorkerError(event.reason);
});

// ============================================================================
// 📦 CACHE MANAGER
// ============================================================================

const cacheManager = {
  async cacheStaticAssets() {
    console.log("📦 Caching critical assets...");
    const cache = await caches.open(CACHE_CONFIG.STATIC);
    return cache.addAll(CRITICAL_ASSETS.map(url => new Request(url, { cache: 'reload' })));
  },

  async cacheCriticalPages() {
    console.log("📄 Pre-caching critical pages...");
    const cache = await caches.open(CACHE_CONFIG.PAGES);
    return Promise.allSettled(
      CRITICAL_PAGES.map(url =>
        cache.add(new Request(url, { cache: 'reload' }))
          .catch(err => console.warn(`⚠️ Failed to cache page ${url}:`, err))
      )
    );
  },

  async cleanupOldCaches() {
    const cacheNames = await caches.keys();
    const oldCaches = cacheNames.filter(name =>
      name.startsWith("disaster-assistant-cache-") && name !== CACHE_NAME
    );
    
    console.log(`🧹 Cleaning up ${oldCaches.length} old caches`);
    return Promise.all(oldCaches.map(cacheName => caches.delete(cacheName)));
  },

  async limitCacheSize(cache, maxItems) {
    const keys = await cache.keys();
    if (keys.length >= maxItems) {
      const toDelete = keys.slice(0, keys.length - maxItems + 1);
      await Promise.all(toDelete.map(key => cache.delete(key)));
    }
  },

  async performMaintenanceCleanup() {
    console.log("🧹 Performing maintenance cleanup...");
    
    const cacheNames = Object.values(CACHE_CONFIG);
    const maxItemsMap = {
      dynamic: CACHE_CONFIG.MAX_DYNAMIC_ITEMS,
      api: CACHE_CONFIG.MAX_API_ITEMS,
      images: CACHE_CONFIG.MAX_IMAGE_ITEMS,
      pages: CACHE_CONFIG.MAX_PAGE_ITEMS
    };
    
    for (const cacheName of cacheNames) {
      const cache = await caches.open(cacheName);
      const maxItems = Object.entries(maxItemsMap).find(([key]) => 
        cacheName.includes(key)
      )?.[1] || 100;
      
      await this.limitCacheSize(cache, maxItems);
    }
    
    await databaseManager.cleanupOldEntries();
    console.log("✅ Maintenance cleanup completed");
  }
};

// ============================================================================
// 🗄️ DATABASE MANAGER
// ============================================================================

const databaseManager = {
  async initialize() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(SW_CONFIG.DB_NAME, SW_CONFIG.DB_VERSION);
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        console.log(`🗄️ Upgrading database schema to version ${SW_CONFIG.DB_VERSION}...`);
        
        Object.values(DB_STORES).forEach(storeName => {
          if (!db.objectStoreNames.contains(storeName)) {
            const store = db.createObjectStore(storeName, { 
              keyPath: "id", 
              autoIncrement: true 
            });
            
            // Add indexes for efficient querying
            store.createIndex("timestamp", "timestamp");
            store.createIndex("priority", "priority");
            store.createIndex("type", "type");
            store.createIndex("status", "status");
            store.createIndex("endpoint", "endpoint");
          }
        });
      };
      
      request.onsuccess = () => {
        console.log("✅ Database initialized");
        resolve(request.result);
      };
      
      request.onerror = () => {
        console.error("❌ Database initialization failed");
        reject(request.error);
      };
    });
  },

  async updateSchema() {
    try {
      await this.initialize();
      console.log("✅ Database schema updated");
    } catch (error) {
      console.error("❌ Database schema update failed:", error);
    }
  },

  async cleanupOldEntries() {
    try {
      const db = await this.initialize();
      const cutoffTime = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days
      
      const analyticsTx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
      const analyticsStore = analyticsTx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
      const analyticsIndex = analyticsStore.index("timestamp");
      
      const oldAnalytics = await analyticsIndex.getAll(IDBKeyRange.upperBound(cutoffTime));
      for (const entry of oldAnalytics) {
        await analyticsStore.delete(entry.id);
      }
      
      console.log(`🗄️ Cleaned up ${oldAnalytics.length} old analytics entries`);
    } catch (error) {
      console.error("❌ Failed to cleanup old database entries:", error);
    }
  }
};

// ============================================================================
// 🔄 SYNC MANAGER
// ============================================================================

const syncManager = {
  getSyncHandler(tag) {
    const syncHandlers = {
      [SYNC_QUEUES.BROADCAST]: () => this.syncQueuedItems(DB_STORES.BROADCAST_QUEUE, "/broadcast"),
      [SYNC_QUEUES.CROWD_REPORT]: () => this.syncQueuedItems(DB_STORES.REPORT_QUEUE, "/api/submit-crowd-report", true),
      [SYNC_QUEUES.TRIAGE]: () => this.syncQueuedItems(DB_STORES.TRIAGE_QUEUE, "/submit-triage", true),
      [SYNC_QUEUES.EMERGENCY_REPORT]: () => this.syncQueuedItems(DB_STORES.EMERGENCY_QUEUE, "/api/submit-emergency-report", true),
      [SYNC_QUEUES.WEATHER_RISK]: () => this.syncQueuedItems(DB_STORES.WEATHER_QUEUE, "/predict-risk"),
      [SYNC_QUEUES.IMAGE_ANALYSIS]: () => this.syncQueuedItems(DB_STORES.IMAGE_QUEUE, "/analyze-image", true),
      [SYNC_QUEUES.FEEDBACK]: () => this.syncQueuedItems(DB_STORES.FEEDBACK_QUEUE, "/api/feedback", true),
      [SYNC_QUEUES.DEVICE_STATUS]: () => this.syncQueuedItems(DB_STORES.DEVICE_STATUS, "/api/device-status"),
      [SYNC_QUEUES.EXPORT_REQUESTS]: () => this.syncQueuedItems(DB_STORES.EXPORT_QUEUE, "/api/export-reports", true)
    };
    
    return syncHandlers[tag];
  },

  async queueForSync(request, syncTag, storeName, options = {}) {
    try {
      const db = await databaseManager.initialize();
      
      let data;
      const contentType = request.headers.get("content-type");
      
      if (contentType?.includes("application/json")) {
        data = await request.json();
      } else if (contentType?.includes("multipart/form-data")) {
        const formData = await request.formData();
        data = {};
        formData.forEach((value, key) => { data[key] = value; });
      } else {
        data = await request.text();
      }
      
      const queueItem = {
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
        status: "pending"
      };
      
      const tx = db.transaction([storeName], "readwrite");
      await tx.objectStore(storeName).add(queueItem);
      
      console.log(`📝 Queued item for sync: ${syncTag}`);
      
      if ("sync" in self.registration) {
        await self.registration.sync.register(syncTag);
      }
      
    } catch (error) {
      console.error("❌ Failed to queue for sync:", error);
    }
  },

  async syncQueuedItems(storeName, endpoint, isFormData = false) {
    try {
      const db = await databaseManager.initialize();
      const tx = db.transaction([storeName], "readwrite");
      const store = tx.objectStore(storeName);
      const items = await store.getAll();
      
      console.log(`🔄 Syncing ${items.length} items from ${storeName}`);
      
      for (const item of items) {
        try {
          let body;
          let headers = { ...item.headers };
          
          if (isFormData) {
            body = new FormData();
            Object.entries(item.data).forEach(([key, value]) => {
              body.append(key, value);
            });
            delete headers["content-type"];
          } else {
            body = JSON.stringify(item.data);
            headers["content-type"] = "application/json";
          }
          
          const response = await utils.fetchWithTimeout(new Request(endpoint, {
            method: item.method,
            headers,
            body
          }));
          
          if (response.ok) {
            await store.delete(item.id);
            console.log(`✅ Successfully synced item ${item.id}`);
          } else {
            await this.handleSyncFailure(store, item);
          }
        } catch (error) {
          console.error(`❌ Error syncing item ${item.id}:`, error);
          await this.handleSyncFailure(store, item);
        }
      }
      
    } catch (error) {
      console.error(`❌ Failed to sync ${storeName}:`, error);
    }
  },

  async handleSyncFailure(store, item) {
    item.retryCount = (item.retryCount || 0) + 1;
    
    if (item.retryCount >= item.maxRetries) {
      console.error(`❌ Max retries exceeded for item ${item.id}, removing from queue`);
      await store.delete(item.id);
    } else {
      item.status = "retrying";
      await store.put(item);
    }
  },

  async checkPendingSyncs() {
    try {
      const db = await databaseManager.initialize();
      const queueStores = Object.values(DB_STORES).filter(store => store.includes('Queue'));
      
      for (const storeName of queueStores) {
        const tx = db.transaction([storeName], "readonly");
        const store = tx.objectStore(storeName);
        const count = await store.count();
        
        if (count > 0) {
          console.log(`📝 Found ${count} pending items in ${storeName}`);
          
          const syncTag = Object.keys(SYNC_QUEUES).find(key =>
            SYNC_QUEUES[key].includes(storeName.replace('Queue', '').toLowerCase())
          );
          
          if (syncTag && "sync" in self.registration) {
            await self.registration.sync.register(SYNC_QUEUES[syncTag]);
          }
        }
      }
    } catch (error) {
      console.error("❌ Failed to check pending syncs:", error);
    }
  },

  async forceSyncAllQueues() {
    console.log("🔄 Force syncing all queues...");
    
    const syncPromises = Object.values(SYNC_QUEUES).map(tag => {
      if ("sync" in self.registration) {
        return self.registration.sync.register(tag);
      }
    });
    
    await Promise.all(syncPromises);
    console.log("✅ All sync queues registered");
  }
};

// ============================================================================
// 🌐 REQUEST HANDLER
// ============================================================================

const requestHandler = {
  handlePostRequest(event, request, urlPath) {
    if (OFFLINE_API_ENDPOINTS.some(endpoint => urlPath.includes(endpoint))) {
      event.respondWith(this.handleOfflineApiRequest(request, urlPath));
      return;
    }
    
    if (urlPath === "/broadcast") {
      event.respondWith(this.handleEmergencyBroadcast(request));
      return;
    }
    
    if (urlPath.includes("/analyze-image") || urlPath.includes("/image-analysis")) {
      event.respondWith(this.handleImageAnalysis(request));
      return;
    }
    
    event.respondWith(this.handleDefaultPost(request));
  },

  handleGetRequest(event, request, urlPath) {
    if (urlPath.startsWith("/static/")) {
      event.respondWith(this.handleStaticAsset(request));
      return;
    }
    
    if (urlPath.startsWith("/api/")) {
      event.respondWith(this.handleApiRequest(request));
      return;
    }
    
    if (CRITICAL_PAGES.includes(urlPath) || urlPath === "/") {
      event.respondWith(this.handleCriticalPageRequest(request));
      return;
    }
    
    event.respondWith(this.handlePageRequest(request));
  },

  async handleOfflineApiRequest(request, urlPath) {
    try {
      const networkResponse = await utils.fetchWithTimeout(request.clone(), 3000);
      
      if (networkResponse.ok) {
        const cache = await caches.open(CACHE_CONFIG.API);
        cache.put(request.url, networkResponse.clone());
      }
      
      return networkResponse;
    } catch (error) {
      console.log(`📡 API ${urlPath} failed, handling offline`);
      
      const { queueType, storeType, priority } = this.getQueueConfigForEndpoint(urlPath);
      
      await syncManager.queueForSync(request.clone(), queueType, storeType, {
        priority,
        endpoint: urlPath,
        timestamp: Date.now()
      });
      
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        console.log(`📦 Serving cached API response for ${urlPath}`);
        return cachedResponse;
      }
      
      return new Response(JSON.stringify({
        status: "queued",
        offline: true,
        priority,
        message: `Request queued for sync. Priority: ${priority}`,
        timestamp: Date.now()
      }), {
        status: 202,
        headers: { "Content-Type": "application/json" }
      });
    }
  },

  async handleEmergencyBroadcast(request) {
    try {
      const networkResponse = await utils.fetchWithTimeout(request.clone());
      
      await syncManager.queueForSync(request.clone(), SYNC_QUEUES.BROADCAST, DB_STORES.BROADCAST_QUEUE, {
        priority: "critical",
        type: "emergency-broadcast"
      });
      
      return networkResponse;
    } catch (error) {
      console.log("📡 Network failed, queuing emergency broadcast");
      
      await syncManager.queueForSync(request.clone(), SYNC_QUEUES.BROADCAST, DB_STORES.BROADCAST_QUEUE, {
        priority: "critical",
        type: "emergency-broadcast"
      });
      
      return new Response(JSON.stringify({
        status: "queued",
        offline: true,
        priority: "critical",
        message: "Emergency broadcast queued for immediate sync when online",
        timestamp: Date.now()
      }), {
        status: 202,
        headers: { "Content-Type": "application/json" }
      });
    }
  },

  async handleImageAnalysis(request) {
    try {
      const response = await utils.fetchWithTimeout(request.clone(), 10000);
      return response;
    } catch (error) {
      console.log("🖼️ Image analysis failed, queuing for later processing");
      
      await syncManager.queueForSync(request.clone(), SYNC_QUEUES.IMAGE_ANALYSIS, DB_STORES.IMAGE_QUEUE, {
        priority: "normal",
        type: "image-analysis"
      });
      
      return new Response(JSON.stringify({
        error: "Image analysis service unavailable",
        queued: true,
        message: "Image will be analyzed when connection is restored"
      }), {
        status: 503,
        headers: { "Content-Type": "application/json" }
      });
    }
  },

  async handleStaticAsset(request) {
    try {
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        this.fetchAndUpdateCache(request);
        return cachedResponse;
      }
      
      const response = await utils.fetchWithTimeout(request);
      
      const cache = await caches.open(
        request.url.includes('/images/') ? CACHE_CONFIG.IMAGES : CACHE_CONFIG.STATIC
      );
      
      if (request.url.includes('/images/')) {
        await cacheManager.limitCacheSize(cache, CACHE_CONFIG.MAX_IMAGE_ITEMS);
      }
      
      cache.put(request, response.clone());
      return response;
      
    } catch (error) {
      if (request.url.includes('/images/')) {
        return fallbackGenerator.generateOfflineImagePlaceholder();
      }
      
      if (request.url.includes('.css')) {
        return fallbackGenerator.generateOfflineCSSFallback();
      }
      
      throw error;
    }
  },

  async handleCriticalPageRequest(request) {
    try {
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        console.log(`📦 Serving critical page from cache: ${request.url}`);
        
        if (navigator.onLine) {
          this.fetchAndUpdateCache(request);
        }
        
        return cachedResponse;
      }
      
      const response = await utils.fetchWithTimeout(request);
      
      const cache = await caches.open(CACHE_CONFIG.PAGES);
      await cacheManager.limitCacheSize(cache, CACHE_CONFIG.MAX_PAGE_ITEMS);
      cache.put(request, response.clone());
      
      return response;
      
    } catch (error) {
      console.log(`📴 Critical page ${request.url} unavailable, serving offline fallback`);
      
      const offlineResponse = await caches.match(OFFLINE_URL);
      return offlineResponse || fallbackGenerator.generateOfflinePageFallback(request.url);
    }
  },

  async handleApiRequest(request) {
    try {
      const response = await utils.fetchWithTimeout(request);
      
      if (response.ok) {
        const cache = await caches.open(CACHE_CONFIG.API);
        await cacheManager.limitCacheSize(cache, CACHE_CONFIG.MAX_API_ITEMS);
        cache.put(request, response.clone());
      }
      
      return response;
    } catch (error) {
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        console.log("📦 Serving API request from cache");
        
        const headers = new Headers(cachedResponse.headers);
        headers.set('X-Served-By', 'ServiceWorker-Cache');
        headers.set('X-Cache-Date', new Date().toISOString());
        
        return new Response(cachedResponse.body, {
          status: cachedResponse.status,
          statusText: cachedResponse.statusText,
          headers
        });
      }
      
      return new Response(JSON.stringify({
        error: "API unavailable offline",
        cached: false,
        endpoint: request.url
      }), {
        status: 503,
        headers: { "Content-Type": "application/json" }
      });
    }
  },

  async handlePageRequest(request) {
    try {
      const response = await utils.fetchWithTimeout(request);
      
      const cache = await caches.open(CACHE_CONFIG.DYNAMIC);
      await cacheManager.limitCacheSize(cache, CACHE_CONFIG.MAX_DYNAMIC_ITEMS);
      cache.put(request, response.clone());
      
      return response;
    } catch (error) {
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        console.log("📦 Serving page from cache");
        return cachedResponse;
      }
      
      console.log("📴 Serving offline page");
      const offlineResponse = await caches.match(OFFLINE_URL);
      return offlineResponse || fallbackGenerator.generateOfflinePageFallback(request.url);
    }
  },

  async handleDefaultPost(request) {
    try {
      return await utils.fetchWithTimeout(request);
    } catch (error) {
      return new Response(JSON.stringify({
        error: "Service unavailable offline",
        queued: false,
        message: "This endpoint is not available offline"
      }), {
        status: 503,
        headers: { "Content-Type": "application/json" }
      });
    }
  },

  async handleOtherMethods(request) {
    try {
      return await utils.fetchWithTimeout(request);
    } catch (error) {
      return new Response(JSON.stringify({
        error: "Service unavailable offline",
        method: request.method,
        message: `${request.method} requests are not supported offline`
      }), {
        status: 503,
        headers: { "Content-Type": "application/json" }
      });
    }
  },

  getQueueConfigForEndpoint(urlPath) {
    if (urlPath.includes("emergency-report") || urlPath.includes("submit-triage")) {
      return {
        queueType: SYNC_QUEUES.EMERGENCY_REPORT,
        storeType: DB_STORES.EMERGENCY_QUEUE,
        priority: "critical"
      };
    }
    
    if (urlPath.includes("crowd-report")) {
      return {
        queueType: SYNC_QUEUES.CROWD_REPORT,
        storeType: DB_STORES.REPORT_QUEUE,
        priority: "high"
      };
    }
    
    if (urlPath.includes("broadcast")) {
      return {
        queueType: SYNC_QUEUES.BROADCAST,
        storeType: DB_STORES.BROADCAST_QUEUE,
        priority: "critical"
      };
    }
    
    if (urlPath.includes("predict-risk") || urlPath.includes("analyze-sentiment")) {
      return {
        queueType: SYNC_QUEUES.WEATHER_RISK,
        storeType: DB_STORES.WEATHER_QUEUE,
        priority: "normal"
      };
    }
    
    if (urlPath.includes("feedback")) {
      return {
        queueType: SYNC_QUEUES.FEEDBACK,
        storeType: DB_STORES.FEEDBACK_QUEUE,
        priority: "normal"
      };
    }
    
    return {
      queueType: SYNC_QUEUES.CROWD_REPORT,
      storeType: DB_STORES.REPORT_QUEUE,
      priority: "normal"
    };
  },

  async fetchAndUpdateCache(request) {
    try {
      const response = await fetch(request);
      if (response.ok) {
        const cache = await caches.open(CACHE_CONFIG.DYNAMIC);
        cache.put(request, response.clone());
      }
    } catch (error) {
      console.log("Background cache update failed:", error);
    }
  }
};

// ============================================================================
// 🎨 FALLBACK GENERATOR
// ============================================================================

const fallbackGenerator = {
  generateOfflineImagePlaceholder() {
    const svg = `
      <svg width="200" height="150" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f3f4f6"/>
        <text x="50%" y="50%" text-anchor="middle" dy="0.3em" fill="#6b7280">
          📱 Offline
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
      body { font-family: Arial, sans-serif; margin: 20px; }
      .offline-notice { background: #fef3c7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
      .offline-mode { border: 2px solid #f59e0b; background: #fffbeb; }
    `;
    
    return new Response(css, {
      headers: {
        'Content-Type': 'text/css',
        'Cache-Control': 'public, max-age=86400'
      }
    });
  },

  generateOfflinePageFallback(url) {
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Offline - Emergency Response</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
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
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            backdrop-filter: blur(10px);
          }
          .offline-icon { font-size: 4rem; margin-bottom: 1rem; }
          .message { 
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
            background: #ef4444;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
          }
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="offline-icon">📱</div>
          <h1><span class="status-indicator"></span>You're Offline</h1>
          <div class="message">
            <p>The page "${url}" is not available offline, but emergency features are still accessible.</p>
            <p><strong>Your reports are being saved locally and will sync when connection is restored.</strong></p>
          </div>
          <div class="actions">
            <a href="/" class="btn">🏠 Home</a>
            <a href="/submit-report" class="btn">🚨 Emergency Report</a>
            <a href="/crowd-reports" class="btn">📊 View Reports</a>
            <a href="/sync-status" class="btn">🔄 Sync Status</a>
          </div>
          <script>
            // Auto-refresh when back online
            window.addEventListener('online', () => {
              document.body.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
              setTimeout(() => window.location.reload(), 1000);
            });
            
            // Update status indicator based on connection
            function updateConnectionStatus() {
              const indicator = document.querySelector('.status-indicator');
              if (navigator.onLine) {
                indicator.style.background = '#10b981';
                indicator.style.animation = 'none';
              } else {
                indicator.style.background = '#ef4444';
                indicator.style.animation = 'pulse 2s infinite';
              }
            }
            
            setInterval(updateConnectionStatus, 1000);
            updateConnectionStatus();
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
  }
};

// ============================================================================
// 📱 NOTIFICATION MANAGER
// ============================================================================

const notificationManager = {
  handlePushNotification(event) {
    console.log("📱 Push notification received");
    
    let notificationData = {
      title: "Emergency Alert",
      body: "Emergency alert received",
      icon: "/static/icons/icon-192x192.png",
      badge: "/static/icons/badge-72x72.png"
    };
    
    if (event.data) {
      try {
        const data = event.data.json();
        notificationData = {
          title: data.title || notificationData.title,
          body: data.body || data.message || notificationData.body,
          icon: data.icon || notificationData.icon,
          badge: data.badge || notificationData.badge,
          data: data,
          actions: [
            { action: "view", title: "View Details", icon: "/static/icons/view.png" },
            { action: "dismiss", title: "Dismiss", icon: "/static/icons/dismiss.png" }
          ],
          vibrate: data.urgent ? [200, 100, 200, 100, 200] : [200, 100, 200],
          requireInteraction: data.urgent || false,
          tag: data.tag || "emergency-alert",
          silent: false,
          timestamp: Date.now()
        };
        
        if (data.urgent) {
          notificationData.requireInteraction = true;
          notificationData.vibrate = [300, 100, 300, 100, 300];
        }
        
      } catch (error) {
        console.error("❌ Failed to parse push notification data:", error);
      }
    }
    
    event.waitUntil(
      Promise.all([
        self.registration.showNotification(notificationData.title, notificationData),
        this.logNotificationReceived(notificationData)
      ])
    );
  },

  handleNotificationClick(event) {
    console.log("📱 Notification clicked:", event.action);
    
    event.notification.close();
    
    const notificationData = event.notification.data || {};
    
    if (event.action === "view") {
      const targetUrl = notificationData.url || notificationData.link || "/";
      
      event.waitUntil(
        clients.matchAll({ type: 'window' }).then(clients => {
          for (const client of clients) {
            if (client.url === targetUrl && 'focus' in client) {
              return client.focus();
            }
          }
          
          if (clients.openWindow) {
            return clients.openWindow(targetUrl);
          }
        })
      );
    } else if (event.action === "dismiss") {
      this.logNotificationAction('dismissed', notificationData);
    } else {
      event.waitUntil(
        clients.matchAll({ type: 'window' }).then(clients => {
          if (clients.length > 0) {
            return clients[0].focus();
          }
          
          if (clients.openWindow) {
            return clients.openWindow("/");
          }
        })
      );
    }
    
    this.logNotificationAction(event.action || 'clicked', notificationData);
  },

  async logNotificationReceived(notificationData) {
    try {
      const db = await databaseManager.initialize();
      const tx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
      const store = tx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
      
      await store.add({
        event: 'notification_received',
        timestamp: Date.now(),
        data: {
          title: notificationData.title,
          tag: notificationData.tag,
          urgent: notificationData.requireInteraction
        }
      });
    } catch (error) {
      console.error("❌ Failed to log notification:", error);
    }
  },

  async logNotificationAction(action, notificationData) {
    try {
      const db = await databaseManager.initialize();
      const tx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
      const store = tx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
      
      await store.add({
        event: 'notification_action',
        timestamp: Date.now(),
        data: {
          action: action,
          tag: notificationData.tag,
          title: notificationData.title
        }
      });
    } catch (error) {
      console.error("❌ Failed to log notification action:", error);
    }
  }
};

// ============================================================================
// 💬 MESSAGE MANAGER
// ============================================================================

const messageManager = {
  handleMessage(event) {
    const { type, data } = event.data || {};
    
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
        
      case "CLEAR_QUEUE":
        event.waitUntil(this.clearSyncQueue(data.queueType));
        break;
        
      case "GET_SYNC_STATUS":
        event.waitUntil(this.sendSyncStatus(event.source));
        break;
        
      default:
        console.warn(`⚠️ Unknown message type: ${type}`);
    }
  },

  async sendOfflineStatus(source) {
    try {
      const db = await databaseManager.initialize();
      const queueCounts = {};
      
      for (const [key, storeName] of Object.entries(DB_STORES)) {
        if (storeName.includes('Queue')) {
          const tx = db.transaction([storeName], "readonly");
          const store = tx.objectStore(storeName);
          queueCounts[key] = await store.count();
        }
      }
      
      source.postMessage({
        type: 'OFFLINE_STATUS',
        data: {
          version: SW_CONFIG.VERSION,
          queueCounts,
          timestamp: Date.now(),
          cacheInfo: await this.getCacheInfo()
        }
      });
    } catch (error) {
      console.error("❌ Failed to get offline status:", error);
    }
  },

  async sendSyncStatus(source) {
    try {
      const db = await databaseManager.initialize();
      const syncStatus = {};
      
      for (const [key, storeName] of Object.entries(DB_STORES)) {
        if (storeName.includes('Queue')) {
          const tx = db.transaction([storeName], "readonly");
          const store = tx.objectStore(storeName);
          const items = await store.getAll();
          
          syncStatus[key] = {
            total: items.length,
            pending: items.filter(item => item.status === 'pending').length,
            retrying: items.filter(item => item.status === 'retrying').length,
            failed: items.filter(item => item.status === 'error').length
          };
        }
      }
      
      source.postMessage({
        type: 'SYNC_STATUS',
        data: {
          syncStatus,
          timestamp: Date.now(),
          online: navigator.onLine
        }
      });
    } catch (error) {
      console.error("❌ Failed to get sync status:", error);
    }
  },

  async getCacheInfo() {
    const cacheNames = await caches.keys();
    const cacheInfo = {};
    
    for (const cacheName of cacheNames) {
      if (cacheName.startsWith('disaster-assistant-cache-')) {
        const cache = await caches.open(cacheName);
        const keys = await cache.keys();
        cacheInfo[cacheName] = keys.length;
      }
    }
    
    return cacheInfo;
  },

  async clearSyncQueue(queueType) {
    try {
      const storeMap = {
        'broadcast': DB_STORES.BROADCAST_QUEUE,
        'crowd-report': DB_STORES.REPORT_QUEUE,
        'triage': DB_STORES.TRIAGE_QUEUE,
        'emergency-report': DB_STORES.EMERGENCY_QUEUE,
        'weather-risk': DB_STORES.WEATHER_QUEUE,
        'image-analysis': DB_STORES.IMAGE_QUEUE,
        'feedback': DB_STORES.FEEDBACK_QUEUE,
        'export-requests': DB_STORES.EXPORT_QUEUE
      };
      
      const storeName = storeMap[queueType];
      if (!storeName) {
        console.warn(`⚠️ Unknown queue type: ${queueType}`);
        return;
      }
      
      const db = await databaseManager.initialize();
      const tx = db.transaction([storeName], "readwrite");
      const store = tx.objectStore(storeName);
      
      await store.clear();
      console.log(`✅ Cleared sync queue: ${queueType}`);
      
    } catch (error) {
      console.error(`❌ Failed to clear sync queue ${queueType}:`, error);
    }
  },

  notifyClientsOfOfflineReadiness() {
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage({
          type: 'OFFLINE_READY',
          version: SW_CONFIG.VERSION,
          timestamp: Date.now(),
          features: {
            caching: true,
            sync: true,
            notifications: true,
            analytics: true
          }
        });
      });
    });
  }
};

// ============================================================================
// 📊 ANALYTICS MANAGER
// ============================================================================

const analyticsManager = {
  async initializeOfflineAnalytics() {
    try {
      const db = await databaseManager.initialize();
      const tx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
      const store = tx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
      
      await store.put({
        id: 'session_start',
        timestamp: Date.now(),
        event: 'sw_installed',
        version: SW_CONFIG.VERSION
      });
      
      console.log("📊 Offline analytics initialized");
    } catch (error) {
      console.error("❌ Failed to initialize offline analytics:", error);
    }
  },

  async logEvent(eventType, eventData = {}) {
    try {
      const db = await databaseManager.initialize();
      const tx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
      const store = tx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
      
      await store.add({
        event: eventType,
        timestamp: Date.now(),
        data: eventData,
        version: SW_CONFIG.VERSION
      });
    } catch (error) {
      console.error("❌ Failed to log analytics event:", error);
    }
  }
};

// ============================================================================
// 🔧 UTILITIES
// ============================================================================

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
      throw error;
    }
  }
};

// ============================================================================
// ❌ ERROR MANAGER
// ============================================================================

const errorManager = {
  async logServiceWorkerError(error) {
    console.error('❌ Service Worker error:', error);
    
    try {
      const db = await databaseManager.initialize();
      const tx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
      const store = tx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
      
      await store.add({
        event: 'sw_error',
        timestamp: Date.now(),
        data: {
          message: error.message || String(error),
          stack: error.stack,
          version: SW_CONFIG.VERSION
        }
      });
    } catch (logError) {
      console.error("❌ Failed to log service worker error:", logError);
    }
  }
};

// ============================================================================
// 🌐 OFFLINE MANAGER
// ============================================================================

const offlineManager = {
  async setupCapabilities() {
    // Set up periodic cache cleanup
    setInterval(() => cacheManager.performMaintenanceCleanup(), 1000 * 60 * 60);
    
    // Set up network status monitoring
    this.setupNetworkMonitoring();
    
    console.log("🔧 Offline capabilities configured");
  },

  setupNetworkMonitoring() {
    let lastOnlineStatus = navigator.onLine;
    
    setInterval(() => {
      const currentOnlineStatus = navigator.onLine;
      
      if (currentOnlineStatus !== lastOnlineStatus) {
        console.log(`🌐 Network status changed: ${currentOnlineStatus ? 'Online' : 'Offline'}`);
        
        if (currentOnlineStatus) {
          syncManager.forceSyncAllQueues();
          analyticsManager.logEvent('network_restored');
        } else {
          analyticsManager.logEvent('network_lost');
        }
        
        // Notify clients of status change
        self.clients.matchAll().then(clients => {
          clients.forEach(client => {
            client.postMessage({
              type: 'NETWORK_STATUS_CHANGED',
              online: currentOnlineStatus,
              timestamp: Date.now()
            });
          });
        });
        
        lastOnlineStatus = currentOnlineStatus;
      }
    }, 5000);
  }
};

// ============================================================================
// 🚀 INITIALIZATION & STARTUP
// ============================================================================

// Perform initial setup
cacheManager.performMaintenanceCleanup();

// Log service worker initialization
console.log(`🚀 Emergency Response Service Worker ${SW_CONFIG.VERSION} initialized`);
console.log(`📦 Caching ${CRITICAL_PAGES.length} critical pages`);
console.log(`🔄 Managing ${Object.keys(SYNC_QUEUES).length} sync queues`);
console.log(`🗄️ Using ${Object.keys(DB_STORES).length} database stores`);
console.log(`🌐 Monitoring ${OFFLINE_API_ENDPOINTS.length} API endpoints`);