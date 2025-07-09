// Enhanced sw.js - Emergency Response Service Worker
const CACHE_VERSION = "v2.2.0";
const CACHE_NAME = `disaster-assistant-cache-${CACHE_VERSION}`;
const OFFLINE_URL = "/offline.html";
const DB_NAME = "EmergencyResponseDB";
const DB_VERSION = 4;

// Enhanced cache configuration
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

// Sync queue configurations
const SYNC_QUEUES = {
  BROADCAST: "emergency-broadcast-sync",
  CROWD_REPORT: "crowd-report-sync",
  TRIAGE: "triage-assessment-sync",
  EMERGENCY_REPORT: "emergency-report-sync",
  WEATHER_RISK: "weather-risk-sync",
  IMAGE_ANALYSIS: "image-analysis-sync"
};

// IndexedDB store configurations
const DB_STORES = {
  BROADCAST_QUEUE: "broadcastQueue",
  REPORT_QUEUE: "crowdReportQueue",
  TRIAGE_QUEUE: "triageQueue",
  EMERGENCY_QUEUE: "emergencyReportQueue",
  WEATHER_QUEUE: "weatherRiskQueue",
  IMAGE_QUEUE: "imageAnalysisQueue",
  SYNC_METADATA: "syncMetadata",
  OFFLINE_ANALYTICS: "offlineAnalytics",
  USER_PREFERENCES: "userPreferences"
};

// ENHANCED: Critical pages that must work offline
const CRITICAL_PAGES = [
  "/",
  "/home",
  "/submit-report",
  "/submit-crowd-report", 
  "/triage-form",
  "/triage-dashboard",
  "/patient-tracker",
  "/patient-list",
  "/crowd-reports",
  "/map-reports",
  "/view-reports",
  "/hazards",
  "/predict",
  "/live-generate",
  "/test-offline",
  "/offline.html"
];

// ENHANCED: Critical assets with fallbacks
const CRITICAL_ASSETS = [
  "/",
  "/offline.html",
  "/static/css/styles.css",
  "/static/js/weather-risk.js",
  "/static/js/sync-queue.js",
  "/static/js/edge-ai.js",
  "/static/js/workers/broadcast.js",
  "/static/js/p2p/fallback-webrtc.js",
  "/static/js/idb.mjs",
  "/manifest.json",
  // ENHANCED: Add Leaflet for offline maps
  "https://unpkg.com/leaflet/dist/leaflet.css",
  "https://unpkg.com/leaflet/dist/leaflet.js",
  // ENHANCED: Add any additional utility JS files
  "/static/js/triage-utils.js",
  "/static/js/pdf-export.js",
  "/static/js/patient-management.js",
  "/static/js/map-utils.js"
];

// ENHANCED: API endpoints that need offline handling
const OFFLINE_API_ENDPOINTS = [
  "/api/submit-crowd-report",
  "/api/submit-emergency-report", 
  "/submit-triage",
  "/predict-risk",
  "/analyze-sentiment",
  "/generate-report",
  "/broadcast",
  "/api/patients",
  "/api/triage-queue",
  "/api/hazard-analysis",
  "/api/sync-status",
  "/api/offline-reports"
];

// Network timeout configuration
const NETWORK_TIMEOUT = 5000; // 5 seconds
const CACHE_TIMEOUT = 1000; // 1 second for cache response

// ============================================================================
// INSTALLATION EVENT - ENHANCED
// ============================================================================
self.addEventListener("install", (event) => {
  console.log("🔧 Service Worker installing...");
  
  event.waitUntil(
    Promise.all([
      // Cache critical assets
      caches.open(CACHE_CONFIG.STATIC).then((cache) => {
        console.log("📦 Caching critical assets...");
        return cache.addAll(CRITICAL_ASSETS.map(url => new Request(url, {
          cache: 'reload' // Force fresh fetch during install
        })));
      }),
      
      // ENHANCED: Pre-cache critical pages
      caches.open(CACHE_CONFIG.PAGES).then((cache) => {
        console.log("📄 Pre-caching critical pages...");
        return Promise.allSettled(
          CRITICAL_PAGES.map(url => 
            cache.add(new Request(url, { cache: 'reload' }))
              .catch(err => console.warn(`⚠️ Failed to cache page ${url}:`, err))
          )
        );
      }),
      
      // Initialize IndexedDB with enhanced schema
      initializeDatabase(),
      
      // ENHANCED: Set up offline analytics
      initializeOfflineAnalytics()
    ]).then(() => {
      console.log("✅ Service Worker installation complete");
    })
  );
  
  // Force activation
  self.skipWaiting();
});

// ============================================================================
// ACTIVATION EVENT - ENHANCED
// ============================================================================
self.addEventListener("activate", (event) => {
  console.log("🚀 Service Worker activating...");
  
  event.waitUntil(
    Promise.all([
      // Clean up old caches
      cleanupOldCaches(),
      
      // Claim all clients
      self.clients.claim(),
      
      // Update database schema if needed
      updateDatabaseSchema(),
      
      // ENHANCED: Initialize offline capabilities
      setupOfflineCapabilities(),
      
      // ENHANCED: Check for pending syncs
      checkPendingSyncs()
    ]).then(() => {
      console.log("✅ Service Worker activation complete");
      
      // ENHANCED: Notify clients that offline mode is ready
      notifyClientsOfOfflineReadiness();
    })
  );
});

// ============================================================================
// ENHANCED FETCH EVENT HANDLER
// ============================================================================
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  const urlPath = url.pathname;
  
  // Skip non-http requests
  if (!request.url.startsWith('http')) return;
  
  // ENHANCED: Handle different request types with better strategies
  if (request.method === "POST") {
    handlePostRequest(event, request, urlPath);
  } else if (request.method === "GET") {
    handleGetRequest(event, request, urlPath);
  } else {
    // Handle other methods (PUT, DELETE, etc.)
    event.respondWith(handleOtherMethods(request));
  }
});

// ============================================================================
// ENHANCED POST REQUEST HANDLER
// ============================================================================
function handlePostRequest(event, request, urlPath) {
  // ENHANCED: Better API endpoint matching
  if (OFFLINE_API_ENDPOINTS.some(endpoint => urlPath.includes(endpoint))) {
    event.respondWith(handleOfflineApiRequest(request, urlPath));
    return;
  }
  
  // Emergency broadcast endpoint
  if (urlPath === "/broadcast") {
    event.respondWith(handleEmergencyBroadcast(request));
    return;
  }
  
  // ENHANCED: Image analysis endpoint
  if (urlPath.includes("/analyze-image") || urlPath.includes("/image-analysis")) {
    event.respondWith(handleImageAnalysis(request));
    return;
  }
  
  // Default POST handler with better error handling
  event.respondWith(handleDefaultPost(request));
}

// ============================================================================
// ENHANCED GET REQUEST HANDLER  
// ============================================================================
function handleGetRequest(event, request, urlPath) {
  // ENHANCED: Static assets with better caching strategy
  if (urlPath.startsWith("/static/")) {
    event.respondWith(handleStaticAssetEnhanced(request));
    return;
  }
  
  // ENHANCED: API requests with offline fallback
  if (urlPath.startsWith("/api/")) {
    event.respondWith(handleApiRequestEnhanced(request));
    return;
  }
  
  // ENHANCED: Page requests with offline-first for critical pages
  if (CRITICAL_PAGES.includes(urlPath) || urlPath === "/") {
    event.respondWith(handleCriticalPageRequest(request));
    return;
  }
  
  // Regular page requests
  event.respondWith(handlePageRequest(request));
}

// ============================================================================
// ENHANCED SPECIFIC REQUEST HANDLERS
// ============================================================================

async function handleOfflineApiRequest(request, urlPath) {
  try {
    // ENHANCED: Try network with shorter timeout for API requests
    const networkResponse = await fetchWithTimeout(request.clone(), 3000);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_CONFIG.API);
      cache.put(request.url, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log(`📡 API ${urlPath} failed, handling offline`);
    
    // ENHANCED: Determine sync queue and priority based on endpoint
    const { queueType, storeType, priority } = getQueueConfigForEndpoint(urlPath);
    
    await queueForSync(request.clone(), queueType, storeType, { 
      priority,
      endpoint: urlPath,
      timestamp: Date.now()
    });
    
    // ENHANCED: Try to return cached response if available
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log(`📦 Serving cached API response for ${urlPath}`);
      return cachedResponse;
    }
    
    return new Response(JSON.stringify({ 
      status: "queued", 
      offline: true,
      priority: priority,
      message: `Request queued for sync. Priority: ${priority}`,
      timestamp: Date.now()
    }), {
      status: 202,
      headers: { "Content-Type": "application/json" }
    });
  }
}

async function handleImageAnalysis(request) {
  try {
    const response = await fetchWithTimeout(request.clone(), 10000); // Longer timeout for image processing
    return response;
  } catch (error) {
    console.log("🖼️ Image analysis failed, queuing for later processing");
    
    await queueForSync(request.clone(), SYNC_QUEUES.IMAGE_ANALYSIS, DB_STORES.IMAGE_QUEUE, {
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
}

async function handleStaticAssetEnhanced(request) {
  // ENHANCED: Cache-first with network fallback and offline alternatives
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      // Serve from cache immediately
      fetchAndUpdateCache(request); // Update cache in background
      return cachedResponse;
    }
    
    const response = await fetchWithTimeout(request);
    
    // Cache static assets aggressively
    const cache = await caches.open(
      request.url.includes('/images/') ? CACHE_CONFIG.IMAGES : CACHE_CONFIG.STATIC
    );
    
    if (request.url.includes('/images/')) {
      await limitCacheSize(cache, CACHE_CONFIG.MAX_IMAGE_ITEMS);
    }
    
    cache.put(request, response.clone());
    return response;
    
  } catch (error) {
    // ENHANCED: Return appropriate offline fallbacks
    if (request.url.includes('/images/')) {
      return generateOfflineImagePlaceholder();
    }
    
    if (request.url.includes('.css')) {
      return generateOfflineCSSFallback();
    }
    
    throw error;
  }
}

async function handleCriticalPageRequest(request) {
  // ENHANCED: Cache-first for critical pages to ensure offline availability
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log(`📦 Serving critical page from cache: ${request.url}`);
      
      // Update cache in background if online
      if (navigator.onLine) {
        fetchAndUpdateCache(request);
      }
      
      return cachedResponse;
    }
    
    // If not in cache, try network
    const response = await fetchWithTimeout(request);
    
    // Cache the response
    const cache = await caches.open(CACHE_CONFIG.PAGES);
    await limitCacheSize(cache, CACHE_CONFIG.MAX_PAGE_ITEMS);
    cache.put(request, response.clone());
    
    return response;
    
  } catch (error) {
    console.log(`📴 Critical page ${request.url} unavailable, serving offline fallback`);
    
    // Return generic offline page for critical pages
    const offlineResponse = await caches.match(OFFLINE_URL);
    return offlineResponse || generateOfflinePageFallback(request.url);
  }
}

// ============================================================================
// ENHANCED UTILITY FUNCTIONS
// ============================================================================

function getQueueConfigForEndpoint(urlPath) {
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
  
  return {
    queueType: SYNC_QUEUES.CROWD_REPORT,
    storeType: DB_STORES.REPORT_QUEUE,
    priority: "normal"
  };
}

async function fetchAndUpdateCache(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_CONFIG.DYNAMIC);
      cache.put(request, response.clone());
    }
  } catch (error) {
    // Silently fail - this is background update
    console.log("Background cache update failed:", error);
  }
}

function generateOfflineImagePlaceholder() {
  // Generate a simple SVG placeholder for missing images
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
}

function generateOfflineCSSFallback() {
  // Basic fallback CSS for offline mode
  const css = `
    body { font-family: Arial, sans-serif; margin: 20px; }
    .offline-notice { background: #fef3c7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
  `;
  
  return new Response(css, {
    headers: {
      'Content-Type': 'text/css',
      'Cache-Control': 'public, max-age=86400'
    }
  });
}

function generateOfflinePageFallback(url) {
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>Offline - Emergency Response</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }
        .container { max-width: 600px; margin: 0 auto; padding: 2rem; }
        .offline-icon { font-size: 4rem; margin-bottom: 1rem; }
        .message { background: #fef3c7; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
        .actions { margin-top: 2rem; }
        .btn { display: inline-block; padding: 0.75rem 1.5rem; background: #3b82f6; color: white; text-decoration: none; border-radius: 8px; margin: 0.5rem; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="offline-icon">📱</div>
        <h1>You're Offline</h1>
        <div class="message">
          <p>The page "${url}" is not available offline, but emergency features are still accessible.</p>
        </div>
        <div class="actions">
          <a href="/" class="btn">🏠 Home</a>
          <a href="/submit-report" class="btn">🚨 Emergency Report</a>
          <a href="/crowd-reports" class="btn">📊 View Reports</a>
        </div>
        <script>
          // Auto-refresh when back online
          window.addEventListener('online', () => {
            window.location.reload();
          });
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

async function initializeOfflineAnalytics() {
  try {
    const db = await initializeDatabase();
    const tx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
    const store = tx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
    
    await store.put({
      id: 'session_start',
      timestamp: Date.now(),
      event: 'sw_installed',
      version: CACHE_VERSION
    });
    
    console.log("📊 Offline analytics initialized");
  } catch (error) {
    console.error("❌ Failed to initialize offline analytics:", error);
  }
}

async function setupOfflineCapabilities() {
  // Set up periodic cache cleanup
  setInterval(performMaintenanceCleanup, 1000 * 60 * 60); // Every hour
  
  // Set up sync retry logic
  setupSyncRetryLogic();
  
  console.log("🔧 Offline capabilities configured");
}

async function checkPendingSyncs() {
  try {
    const db = await initializeDatabase();
    const stores = Object.values(DB_STORES).filter(store => store.includes('Queue'));
    
    for (const storeName of stores) {
      const tx = db.transaction([storeName], "readonly");
      const store = tx.objectStore(storeName);
      const count = await store.count();
      
      if (count > 0) {
        console.log(`📝 Found ${count} pending items in ${storeName}`);
        
        // Register appropriate sync
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
}

function notifyClientsOfOfflineReadiness() {
  self.clients.matchAll().then(clients => {
    clients.forEach(client => {
      client.postMessage({
        type: 'OFFLINE_READY',
        version: CACHE_VERSION,
        timestamp: Date.now()
      });
    });
  });
}

function setupSyncRetryLogic() {
  // Enhanced retry logic with exponential backoff
  self.addEventListener('sync', (event) => {
    console.log(`🔄 Background sync triggered: ${event.tag}`);
    
    const syncHandlers = {
      [SYNC_QUEUES.BROADCAST]: () => syncQueuedItems(DB_STORES.BROADCAST_QUEUE, "/broadcast"),
      [SYNC_QUEUES.CROWD_REPORT]: () => syncQueuedItems(DB_STORES.REPORT_QUEUE, "/api/submit-crowd-report", true),
      [SYNC_QUEUES.TRIAGE]: () => syncQueuedItems(DB_STORES.TRIAGE_QUEUE, "/submit-triage", true),
      [SYNC_QUEUES.EMERGENCY_REPORT]: () => syncQueuedItems(DB_STORES.EMERGENCY_QUEUE, "/api/submit-emergency-report", true),
      [SYNC_QUEUES.WEATHER_RISK]: () => syncQueuedItems(DB_STORES.WEATHER_QUEUE, "/predict-risk"),
      [SYNC_QUEUES.IMAGE_ANALYSIS]: () => syncQueuedItems(DB_STORES.IMAGE_QUEUE, "/analyze-image", true)
    };
    
    const handler = syncHandlers[event.tag];
    if (handler) {
      event.waitUntil(handler());
    } else {
      console.warn(`⚠️ Unknown sync tag: ${event.tag}`);
    }
  });
}

// ============================================================================
// ENHANCED DATABASE FUNCTIONS
// ============================================================================

async function initializeDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      console.log(`🗄️ Upgrading database schema to version ${DB_VERSION}...`);
      
      // Create all required object stores
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
}

// ============================================================================
// ENHANCED MESSAGE HANDLING
// ============================================================================

self.addEventListener("message", (event) => {
  const { type, data } = event.data || {};
  
  switch (type) {
    case "CLEANUP_CACHES":
      event.waitUntil(performMaintenanceCleanup());
      break;
      
    case "FORCE_SYNC":
      event.waitUntil(forceSyncAllQueues());
      break;
      
    case "GET_OFFLINE_STATUS":
      event.waitUntil(sendOfflineStatus(event.source));
      break;
      
    case "CLEAR_QUEUE":
      event.waitUntil(clearSyncQueue(data.queueType));
      break;
      
    default:
      console.warn(`⚠️ Unknown message type: ${type}`);
  }
});

async function forceSyncAllQueues() {
  console.log("🔄 Force syncing all queues...");
  
  const syncPromises = Object.entries(SYNC_QUEUES).map(([key, tag]) => {
    if ("sync" in self.registration) {
      return self.registration.sync.register(tag);
    }
  });
  
  await Promise.all(syncPromises);
  console.log("✅ All sync queues registered");
}

async function sendOfflineStatus(source) {
  try {
    const db = await initializeDatabase();
    const queueCounts = {};
    
    // Get queue counts
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
        version: CACHE_VERSION,
        queueCounts,
        timestamp: Date.now()
      }
    });
  } catch (error) {
    console.error("❌ Failed to get offline status:", error);
  }
}

// ============================================================================
// Keep all your existing utility functions...
// ============================================================================

// [Previous utility functions remain the same]
async function cleanupOldCaches() {
  const cacheNames = await caches.keys();
  const oldCaches = cacheNames.filter(name => 
    name.startsWith("disaster-assistant-cache-") && name !== CACHE_NAME
  );
  
  console.log(`🧹 Cleaning up ${oldCaches.length} old caches`);
  
  return Promise.all(
    oldCaches.map(cacheName => caches.delete(cacheName))
  );
}

async function queueForSync(request, syncTag, storeName, options = {}) {
  try {
    const db = await initializeDatabase();
    
    let data;
    const contentType = request.headers.get("content-type");
    
    if (contentType && contentType.includes("application/json")) {
      data = await request.json();
    } else if (contentType && contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      data = {};
      formData.forEach((value, key) => {
        data[key] = value;
      });
    } else {
      data = await request.text();
    }
    
    const queueItem = {
      data: data,
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
    
    // Register for background sync
    if ("sync" in self.registration) {
      await self.registration.sync.register(syncTag);
    }
    
  } catch (error) {
    console.error("❌ Failed to queue for sync:", error);
  }
}

async function syncQueuedItems(storeName, endpoint, isFormData = false) {
  try {
    const db = await initializeDatabase();
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
        
        const response = await fetchWithTimeout(new Request(endpoint, {
          method: item.method,
          headers: headers,
          body: body
        }));
        
        if (response.ok) {
          await store.delete(item.id);
          console.log(`✅ Successfully synced item ${item.id}`);
        } else {
          console.warn(`⚠️ Sync failed for item ${item.id}: ${response.status}`);
          
          item.retryCount = (item.retryCount || 0) + 1;
          
          if (item.retryCount >= item.maxRetries) {
            console.error(`❌ Max retries exceeded for item ${item.id}, removing from queue`);
            await store.delete(item.id);
          } else {
            item.status = "retrying";
            await store.put(item);
          }
        }
      } catch (error) {
        console.error(`❌ Error syncing item ${item.id}:`, error);
        
        item.retryCount = (item.retryCount || 0) + 1;
        
        if (item.retryCount >= item.maxRetries) {
          console.error(`❌ Max retries exceeded for item ${item.id}, removing from queue`);
          await store.delete(item.id);
        } else {
          item.status = "error";
          await store.put(item);
        }
      }
    }
    
  } catch (error) {
    console.error(`❌ Failed to sync ${storeName}:`, error);
  }
}

async function fetchWithTimeout(request, timeout = NETWORK_TIMEOUT) {
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

async function limitCacheSize(cache, maxItems) {
  const keys = await cache.keys();
  if (keys.length >= maxItems) {
    const toDelete = keys.slice(0, keys.length - maxItems + 1);
    await Promise.all(toDelete.map(key => cache.delete(key)));
  }
}

async function performMaintenanceCleanup() {
  console.log("🧹 Performing maintenance cleanup...");
  
  const cacheNames = Object.values(CACHE_CONFIG);
  
  for (const cacheName of cacheNames) {
    const cache = await caches.open(cacheName);
    const maxItems = cacheName.includes("dynamic") ? CACHE_CONFIG.MAX_DYNAMIC_ITEMS :
                     cacheName.includes("api") ? CACHE_CONFIG.MAX_API_ITEMS :
                     cacheName.includes("images") ? CACHE_CONFIG.MAX_IMAGE_ITEMS :
                     cacheName.includes("pages") ? CACHE_CONFIG.MAX_PAGE_ITEMS : 100;
    
    await limitCacheSize(cache, maxItems);
  }
  
  // Clean up old database entries
  await cleanupOldDatabaseEntries();
  
  console.log("✅ Maintenance cleanup completed");
}

async function cleanupOldDatabaseEntries() {
  try {
    const db = await initializeDatabase();
    const cutoffTime = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days ago
    
    // Clean up old analytics entries
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

async function updateDatabaseSchema() {
  // Handle any database schema updates needed for version changes
  try {
    const db = await initializeDatabase();
    console.log("✅ Database schema updated");
  } catch (error) {
    console.error("❌ Database schema update failed:", error);
  }
}

async function handleDefaultPost(request) {
  try {
    return await fetchWithTimeout(request);
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
}

async function handleOtherMethods(request) {
  try {
    return await fetchWithTimeout(request);
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
}

async function handleApiRequestEnhanced(request) {
  try {
    const response = await fetchWithTimeout(request);
    
    // Cache successful API responses
    if (response.ok) {
      const cache = await caches.open(CACHE_CONFIG.API);
      await limitCacheSize(cache, CACHE_CONFIG.MAX_API_ITEMS);
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    // Try to serve from cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log("📦 Serving API request from cache");
      
      // Add cache headers to indicate offline response
      const headers = new Headers(cachedResponse.headers);
      headers.set('X-Served-By', 'ServiceWorker-Cache');
      headers.set('X-Cache-Date', new Date().toISOString());
      
      return new Response(cachedResponse.body, {
        status: cachedResponse.status,
        statusText: cachedResponse.statusText,
        headers: headers
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
}

async function handlePageRequest(request) {
  try {
    const response = await fetchWithTimeout(request);
    
    // Cache successful page responses
    const cache = await caches.open(CACHE_CONFIG.DYNAMIC);
    await limitCacheSize(cache, CACHE_CONFIG.MAX_DYNAMIC_ITEMS);
    cache.put(request, response.clone());
    
    return response;
  } catch (error) {
    // Try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log("📦 Serving page from cache");
      return cachedResponse;
    }
    
    // Fallback to offline page
    console.log("📴 Serving offline page");
    const offlineResponse = await caches.match(OFFLINE_URL);
    return offlineResponse || generateOfflinePageFallback(request.url);
  }
}

async function handleEmergencyBroadcast(request) {
  try {
    // Try network first for immediate broadcast
    const networkResponse = await fetchWithTimeout(request.clone());
    
    // Also queue for reliability
    await queueForSync(request.clone(), SYNC_QUEUES.BROADCAST, DB_STORES.BROADCAST_QUEUE, { 
      priority: "critical",
      type: "emergency-broadcast"
    });
    
    return networkResponse;
  } catch (error) {
    console.log("📡 Network failed, queuing emergency broadcast");
    
    await queueForSync(request.clone(), SYNC_QUEUES.BROADCAST, DB_STORES.BROADCAST_QUEUE, { 
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
}

async function clearSyncQueue(queueType) {
  try {
    const storeMap = {
      'broadcast': DB_STORES.BROADCAST_QUEUE,
      'crowd-report': DB_STORES.REPORT_QUEUE,
      'triage': DB_STORES.TRIAGE_QUEUE,
      'emergency-report': DB_STORES.EMERGENCY_QUEUE,
      'weather-risk': DB_STORES.WEATHER_QUEUE,
      'image-analysis': DB_STORES.IMAGE_QUEUE
    };
    
    const storeName = storeMap[queueType];
    if (!storeName) {
      console.warn(`⚠️ Unknown queue type: ${queueType}`);
      return;
    }
    
    const db = await initializeDatabase();
    const tx = db.transaction([storeName], "readwrite");
    const store = tx.objectStore(storeName);
    
    await store.clear();
    console.log(`✅ Cleared sync queue: ${queueType}`);
    
  } catch (error) {
    console.error(`❌ Failed to clear sync queue ${queueType}:`, error);
  }
}

// ============================================================================
// ENHANCED PUSH NOTIFICATION HANDLING
// ============================================================================
self.addEventListener("push", (event) => {
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
      
      // Add urgency-based styling
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
      logNotificationReceived(notificationData)
    ])
  );
});

async function logNotificationReceived(notificationData) {
  try {
    const db = await initializeDatabase();
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
}

// ============================================================================
// ENHANCED NOTIFICATION CLICK HANDLING
// ============================================================================
self.addEventListener("notificationclick", (event) => {
  console.log("📱 Notification clicked:", event.action);
  
  event.notification.close();
  
  const notificationData = event.notification.data || {};
  
  if (event.action === "view") {
    const targetUrl = notificationData.url || notificationData.link || "/";
    
    event.waitUntil(
      clients.matchAll({ type: 'window' }).then(clients => {
        // Check if there's already a window open
        for (const client of clients) {
          if (client.url === targetUrl && 'focus' in client) {
            return client.focus();
          }
        }
        
        // Open new window
        if (clients.openWindow) {
          return clients.openWindow(targetUrl);
        }
      })
    );
  } else if (event.action === "dismiss") {
    // Log dismissal
    logNotificationAction('dismissed', notificationData);
  } else {
    // Default click action - open main app
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
  
  // Log the interaction
  logNotificationAction(event.action || 'clicked', notificationData);
});

async function logNotificationAction(action, notificationData) {
  try {
    const db = await initializeDatabase();
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

// ============================================================================
// PERIODIC BACKGROUND TASKS
// ============================================================================

// Set up periodic cleanup when SW starts
performMaintenanceCleanup();

// Log service worker version and capabilities
console.log(`🚀 Emergency Response Service Worker ${CACHE_VERSION} initialized`);
console.log(`📦 Caching ${CRITICAL_PAGES.length} critical pages`);
console.log(`🔄 Managing ${Object.keys(SYNC_QUEUES).length} sync queues`);
console.log(`🗄️ Using ${Object.keys(DB_STORES).length} database stores`);

// ============================================================================
// ERROR HANDLING AND RECOVERY
// ============================================================================

self.addEventListener('error', (event) => {
  console.error('❌ Service Worker error:', event.error);
  
  // Log error for diagnostics
  logServiceWorkerError(event.error);
});

self.addEventListener('unhandledrejection', (event) => {
  console.error('❌ Service Worker unhandled rejection:', event.reason);
  
  // Log error for diagnostics  
  logServiceWorkerError(event.reason);
});

async function logServiceWorkerError(error) {
  try {
    const db = await initializeDatabase();
    const tx = db.transaction([DB_STORES.OFFLINE_ANALYTICS], "readwrite");
    const store = tx.objectStore(DB_STORES.OFFLINE_ANALYTICS);
    
    await store.add({
      event: 'sw_error',
      timestamp: Date.now(),
      data: {
        message: error.message || String(error),
        stack: error.stack,
        version: CACHE_VERSION
      }
    });
  } catch (logError) {
    console.error("❌ Failed to log service worker error:", logError);
  }
}

// ============================================================================
// NETWORK CONNECTIVITY MONITORING
// ============================================================================

let lastOnlineStatus = navigator.onLine;

setInterval(() => {
  const currentOnlineStatus = navigator.onLine;
  
  if (currentOnlineStatus !== lastOnlineStatus) {
    console.log(`🌐 Network status changed: ${currentOnlineStatus ? 'Online' : 'Offline'}`);
    
    if (currentOnlineStatus) {
      // Back online - trigger sync for all queues
      forceSyncAllQueues();
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
}, 5000); // Check every 5 seconds