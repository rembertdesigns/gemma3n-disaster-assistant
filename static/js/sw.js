// Enhanced sw.js - Emergency Response Service Worker
const CACHE_VERSION = "v2.1.0";
const CACHE_NAME = `disaster-assistant-cache-${CACHE_VERSION}`;
const OFFLINE_URL = "/offline.html";
const DB_NAME = "EmergencyResponseDB";
const DB_VERSION = 3;

// Enhanced cache configuration
const CACHE_CONFIG = {
  STATIC: `${CACHE_NAME}-static`,
  DYNAMIC: `${CACHE_NAME}-dynamic`,
  API: `${CACHE_NAME}-api`,
  IMAGES: `${CACHE_NAME}-images`,
  MAX_DYNAMIC_ITEMS: 50,
  MAX_API_ITEMS: 100,
  MAX_IMAGE_ITEMS: 30
};

// Sync queue configurations
const SYNC_QUEUES = {
  BROADCAST: "emergency-broadcast-sync",
  CROWD_REPORT: "crowd-report-sync",
  TRIAGE: "triage-assessment-sync",
  EMERGENCY_REPORT: "emergency-report-sync",
  WEATHER_RISK: "weather-risk-sync"
};

// IndexedDB store configurations
const DB_STORES = {
  BROADCAST_QUEUE: "broadcastQueue",
  REPORT_QUEUE: "crowdReportQueue",
  TRIAGE_QUEUE: "triageQueue",
  EMERGENCY_QUEUE: "emergencyReportQueue",
  WEATHER_QUEUE: "weatherRiskQueue",
  SYNC_METADATA: "syncMetadata",
  OFFLINE_ANALYTICS: "offlineAnalytics"
};

// Critical assets that must be cached for offline operation
const CRITICAL_ASSETS = [
  "/",
  "/offline.html",
  "/static/css/styles.css",
  "/static/js/weather-risk.js",
  "/static/js/sync-queue.js",
  "/static/js/edge-ai.js",
  "/static/js/workers/broadcast.js",
  "/static/js/p2p/fallback-webrtc.js",
  "/manifest.json"
];

// Additional assets to cache opportunistically
const OPTIONAL_ASSETS = [
  "/predict",
  "/submit-report",
  "/submit-crowd-report",
  "/triage-form",
  "/crowd-reports",
  "/map-reports",
  "/static/mock_hazard_image.jpg",
  "/static/js/workers/broadcast-map.js",
  "/static/js/idb.mjs"
];

// Network timeout configuration
const NETWORK_TIMEOUT = 5000; // 5 seconds
const CACHE_TIMEOUT = 1000; // 1 second for cache response

// ============================================================================
// INSTALLATION EVENT
// ============================================================================
self.addEventListener("install", (event) => {
  console.log("🔧 Service Worker installing...");
  
  event.waitUntil(
    Promise.all([
      // Cache critical assets
      caches.open(CACHE_CONFIG.STATIC).then((cache) => {
        console.log("📦 Caching critical assets...");
        return cache.addAll(CRITICAL_ASSETS);
      }),
      
      // Cache optional assets (don't fail if some are missing)
      caches.open(CACHE_CONFIG.DYNAMIC).then((cache) => {
        console.log("📦 Caching optional assets...");
        return Promise.allSettled(
          OPTIONAL_ASSETS.map(url => cache.add(url).catch(err => {
            console.warn(`⚠️ Failed to cache ${url}:`, err);
          }))
        );
      }),
      
      // Initialize IndexedDB
      initializeDatabase()
    ]).then(() => {
      console.log("✅ Service Worker installation complete");
    })
  );
  
  // Force activation
  self.skipWaiting();
});

// ============================================================================
// ACTIVATION EVENT
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
      updateDatabaseSchema()
    ]).then(() => {
      console.log("✅ Service Worker activation complete");
    })
  );
});

// ============================================================================
// FETCH EVENT HANDLER
// ============================================================================
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  const urlPath = url.pathname;
  
  // Handle different request types
  if (request.method === "POST") {
    handlePostRequest(event, request, urlPath);
  } else if (request.method === "GET") {
    handleGetRequest(event, request, urlPath);
  }
});

// ============================================================================
// POST REQUEST HANDLER
// ============================================================================
function handlePostRequest(event, request, urlPath) {
  // Emergency broadcast endpoint
  if (urlPath === "/broadcast") {
    event.respondWith(handleEmergencyBroadcast(request));
    return;
  }
  
  // Crowd report submission
  if (urlPath === "/api/submit-crowd-report") {
    event.respondWith(handleCrowdReportSubmission(request));
    return;
  }
  
  // Emergency report submission
  if (urlPath === "/api/submit-emergency-report" || urlPath === "/submit-triage") {
    event.respondWith(handleEmergencyReportSubmission(request, urlPath));
    return;
  }
  
  // Weather risk prediction
  if (urlPath === "/predict-risk" || urlPath === "/analyze-sentiment") {
    event.respondWith(handleWeatherRiskRequest(request, urlPath));
    return;
  }
  
  // PDF generation
  if (urlPath === "/generate-report") {
    event.respondWith(handleReportGeneration(request));
    return;
  }
  
  // Default POST handler
  event.respondWith(
    fetchWithTimeout(request).catch(() => 
      new Response(JSON.stringify({ 
        error: "Service unavailable offline",
        queued: false 
      }), {
        status: 503,
        headers: { "Content-Type": "application/json" }
      })
    )
  );
}

// ============================================================================
// GET REQUEST HANDLER
// ============================================================================
function handleGetRequest(event, request, urlPath) {
  // API requests
  if (urlPath.startsWith("/api/")) {
    event.respondWith(handleApiRequest(request));
    return;
  }
  
  // Static assets
  if (urlPath.startsWith("/static/")) {
    event.respondWith(handleStaticAsset(request));
    return;
  }
  
  // Page requests
  event.respondWith(handlePageRequest(request));
}

// ============================================================================
// SPECIFIC REQUEST HANDLERS
// ============================================================================

async function handleEmergencyBroadcast(request) {
  try {
    // Try network first for immediate broadcast
    const networkResponse = await fetchWithTimeout(request.clone());
    
    // Also queue for reliability
    await queueForSync(request.clone(), SYNC_QUEUES.BROADCAST, DB_STORES.BROADCAST_QUEUE);
    
    return networkResponse;
  } catch (error) {
    console.log("📡 Network failed, queuing emergency broadcast");
    
    await queueForSync(request.clone(), SYNC_QUEUES.BROADCAST, DB_STORES.BROADCAST_QUEUE, { priority: "critical" });
    
    return new Response(JSON.stringify({ 
      status: "queued", 
      offline: true,
      priority: "critical",
      message: "Emergency broadcast queued for immediate sync when online"
    }), {
      status: 202,
      headers: { "Content-Type": "application/json" }
    });
  }
}

async function handleCrowdReportSubmission(request) {
  try {
    const response = await fetchWithTimeout(request.clone());
    return response;
  } catch (error) {
    console.log("📝 Network failed, queuing crowd report");
    
    await queueForSync(request.clone(), SYNC_QUEUES.CROWD_REPORT, DB_STORES.REPORT_QUEUE);
    
    return new Response(JSON.stringify({ 
      status: "queued", 
      offline: true,
      message: "Report saved locally and will be submitted when connection is restored"
    }), {
      status: 202,
      headers: { "Content-Type": "application/json" }
    });
  }
}

async function handleEmergencyReportSubmission(request, urlPath) {
  try {
    const response = await fetchWithTimeout(request.clone());
    return response;
  } catch (error) {
    console.log("🚨 Network failed, queuing emergency report");
    
    const queueType = urlPath.includes("triage") ? SYNC_QUEUES.TRIAGE : SYNC_QUEUES.EMERGENCY_REPORT;
    const storeType = urlPath.includes("triage") ? DB_STORES.TRIAGE_QUEUE : DB_STORES.EMERGENCY_QUEUE;
    
    await queueForSync(request.clone(), queueType, storeType, { priority: "high" });
    
    return new Response(JSON.stringify({ 
      status: "queued", 
      offline: true,
      priority: "high",
      message: "Emergency report saved locally with high priority for sync"
    }), {
      status: 202,
      headers: { "Content-Type": "application/json" }
    });
  }
}

async function handleWeatherRiskRequest(request, urlPath) {
  try {
    const response = await fetchWithTimeout(request.clone());
    
    // Cache successful risk predictions
    if (response.ok && urlPath === "/predict-risk") {
      const cache = await caches.open(CACHE_CONFIG.API);
      cache.put(request.url + "?" + Date.now(), response.clone());
    }
    
    return response;
  } catch (error) {
    console.log("🌦️ Network failed, using cached weather data or queuing request");
    
    if (urlPath === "/predict-risk") {
      // Try to return cached prediction
      const cachedResponse = await getCachedWeatherPrediction();
      if (cachedResponse) {
        return cachedResponse;
      }
    }
    
    // Queue for later processing
    await queueForSync(request.clone(), SYNC_QUEUES.WEATHER_RISK, DB_STORES.WEATHER_QUEUE);
    
    return new Response(JSON.stringify({ 
      error: "Weather service unavailable",
      cached: false,
      queued: true
    }), {
      status: 503,
      headers: { "Content-Type": "application/json" }
    });
  }
}

async function handleReportGeneration(request) {
  try {
    return await fetchWithTimeout(request.clone());
  } catch (error) {
    console.log("📄 Network failed, cannot generate PDF offline");
    
    return new Response(JSON.stringify({ 
      error: "PDF generation requires internet connection",
      message: "Report data has been saved. PDF will be available when connection is restored."
    }), {
      status: 503,
      headers: { "Content-Type": "application/json" }
    });
  }
}

async function handleApiRequest(request) {
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
      return cachedResponse;
    }
    
    return new Response(JSON.stringify({ 
      error: "API unavailable offline",
      cached: false
    }), {
      status: 503,
      headers: { "Content-Type": "application/json" }
    });
  }
}

async function handleStaticAsset(request) {
  // Cache-first strategy for static assets
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const response = await fetchWithTimeout(request);
    
    // Cache static assets
    const cache = await caches.open(
      request.url.includes('/images/') ? CACHE_CONFIG.IMAGES : CACHE_CONFIG.STATIC
    );
    
    if (request.url.includes('/images/')) {
      await limitCacheSize(cache, CACHE_CONFIG.MAX_IMAGE_ITEMS);
    }
    
    cache.put(request, response.clone());
    
    return response;
  } catch (error) {
    // Return offline fallback for images
    if (request.url.includes('/images/')) {
      return new Response('', { status: 200, statusText: 'Offline' });
    }
    
    throw error;
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
    return caches.match(OFFLINE_URL);
  }
}

// ============================================================================
// BACKGROUND SYNC EVENT
// ============================================================================
self.addEventListener("sync", (event) => {
  console.log(`🔄 Background sync triggered: ${event.tag}`);
  
  switch (event.tag) {
    case SYNC_QUEUES.BROADCAST:
      event.waitUntil(syncQueuedItems(DB_STORES.BROADCAST_QUEUE, "/broadcast"));
      break;
    case SYNC_QUEUES.CROWD_REPORT:
      event.waitUntil(syncQueuedItems(DB_STORES.REPORT_QUEUE, "/api/submit-crowd-report", true));
      break;
    case SYNC_QUEUES.TRIAGE:
      event.waitUntil(syncQueuedItems(DB_STORES.TRIAGE_QUEUE, "/submit-triage", true));
      break;
    case SYNC_QUEUES.EMERGENCY_REPORT:
      event.waitUntil(syncQueuedItems(DB_STORES.EMERGENCY_QUEUE, "/api/submit-emergency-report", true));
      break;
    case SYNC_QUEUES.WEATHER_RISK:
      event.waitUntil(syncQueuedItems(DB_STORES.WEATHER_QUEUE, "/predict-risk"));
      break;
    default:
      console.warn(`⚠️ Unknown sync tag: ${event.tag}`);
  }
});

// ============================================================================
// PUSH NOTIFICATION EVENT
// ============================================================================
self.addEventListener("push", (event) => {
  console.log("📱 Push notification received");
  
  const options = {
    body: event.data ? event.data.text() : "Emergency alert received",
    icon: "/static/icons/icon-192x192.png",
    badge: "/static/icons/badge-72x72.png",
    vibrate: [200, 100, 200],
    data: event.data ? JSON.parse(event.data.text()) : {},
    actions: [
      { action: "view", title: "View Details" },
      { action: "dismiss", title: "Dismiss" }
    ],
    requireInteraction: true,
    tag: "emergency-alert"
  };
  
  event.waitUntil(
    self.registration.showNotification("Emergency Alert", options)
  );
});

// ============================================================================
// NOTIFICATION CLICK EVENT
// ============================================================================
self.addEventListener("notificationclick", (event) => {
  console.log("📱 Notification clicked:", event.action);
  
  event.notification.close();
  
  if (event.action === "view") {
    event.waitUntil(
      clients.openWindow("/") // or specific emergency page
    );
  }
});

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

async function initializeDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      console.log("🗄️ Upgrading database schema...");
      
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

async function updateDatabaseSchema() {
  // Placeholder for future schema updates
  console.log("🔄 Checking database schema...");
}

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
      retryCount: 0,
      maxRetries: options.maxRetries || 3
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
          // Remove content-type header to let browser set it with boundary
          delete headers["content-type"];
        } else {
          body = JSON.stringify(item.data);
          headers["content-type"] = "application/json";
        }
        
        const response = await fetch(endpoint, {
          method: item.method,
          headers: headers,
          body: body
        });
        
        if (response.ok) {
          await store.delete(item.id);
          console.log(`✅ Successfully synced item ${item.id}`);
        } else {
          console.warn(`⚠️ Sync failed for item ${item.id}: ${response.status}`);
          
          // Increment retry count
          item.retryCount = (item.retryCount || 0) + 1;
          
          if (item.retryCount >= item.maxRetries) {
            console.error(`❌ Max retries exceeded for item ${item.id}, removing from queue`);
            await store.delete(item.id);
          } else {
            await store.put(item);
          }
        }
      } catch (error) {
        console.error(`❌ Error syncing item ${item.id}:`, error);
        
        // Increment retry count on network errors too
        item.retryCount = (item.retryCount || 0) + 1;
        
        if (item.retryCount >= item.maxRetries) {
          console.error(`❌ Max retries exceeded for item ${item.id}, removing from queue`);
          await store.delete(item.id);
        } else {
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
    // Remove oldest entries
    const toDelete = keys.slice(0, keys.length - maxItems + 1);
    await Promise.all(toDelete.map(key => cache.delete(key)));
  }
}

async function getCachedWeatherPrediction() {
  try {
    const cache = await caches.open(CACHE_CONFIG.API);
    const keys = await cache.keys();
    
    // Find the most recent weather prediction
    const weatherKeys = keys.filter(key => key.url.includes("/predict-risk"));
    
    if (weatherKeys.length > 0) {
      const latestKey = weatherKeys[weatherKeys.length - 1];
      const response = await cache.match(latestKey);
      
      if (response) {
        console.log("🌦️ Serving cached weather prediction");
        
        // Add cache indicator to response
        const data = await response.json();
        data._cached = true;
        data._cacheTimestamp = Date.now();
        
        return new Response(JSON.stringify(data), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        });
      }
    }
  } catch (error) {
    console.error("❌ Error retrieving cached weather prediction:", error);
  }
  
  return null;
}

// ============================================================================
// PERIODIC CLEANUP
// ============================================================================
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "CLEANUP_CACHES") {
    event.waitUntil(performMaintenanceCleanup());
  }
});

async function performMaintenanceCleanup() {
  console.log("🧹 Performing maintenance cleanup...");
  
  // Clean up old cache entries
  const cacheNames = Object.values(CACHE_CONFIG);
  
  for (const cacheName of cacheNames) {
    const cache = await caches.open(cacheName);
    const maxItems = cacheName.includes("dynamic") ? CACHE_CONFIG.MAX_DYNAMIC_ITEMS :
                     cacheName.includes("api") ? CACHE_CONFIG.MAX_API_ITEMS :
                     cacheName.includes("images") ? CACHE_CONFIG.MAX_IMAGE_ITEMS : 100;
    
    await limitCacheSize(cache, maxItems);
  }
  
  console.log("✅ Maintenance cleanup completed");
}

// Log service worker version
console.log(`🚀 Emergency Response Service Worker ${CACHE_VERSION} initialized`);