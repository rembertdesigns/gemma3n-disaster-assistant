const CACHE_NAME = "disaster-assistant-cache-v1";
const OFFLINE_URL = "/offline.html";
const BROADCAST_QUEUE = "broadcast-sync-queue";
const DB_NAME = "DisasterBroadcastsDB";
const STORE_NAME = "broadcastQueue";

const ASSETS_TO_CACHE = [
  "/",
  OFFLINE_URL,
  "/static/css/styles.css",
  "/static/js/weather-risk.js",
  "/static/js/workers/broadcast.js",
  "/static/js/p2p/fallback-webrtc.js",
  "/static/mock_hazard_image.jpg"
];

// -------- Install
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS_TO_CACHE))
  );
  self.skipWaiting();
});

// -------- Activate
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.map((key) => key !== CACHE_NAME && caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

// -------- Fetch Handler
self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Handle POST to /broadcast for offline fallback
  if (
    request.method === "POST" &&
    new URL(request.url).pathname === "/broadcast"
  ) {
    event.respondWith(
      fetch(request.clone()).catch(() => {
        return queueBroadcast(request.clone()).then(() => {
          return new Response(
            JSON.stringify({ status: "queued", offline: true }),
            { headers: { "Content-Type": "application/json" } }
          );
        });
      })
    );
    return;
  }

  // Handle GET requests: cache first, fallback to network
  event.respondWith(
    fetch(request).catch(() =>
      caches.match(request).then((res) => res || caches.match(OFFLINE_URL))
    )
  );
});

// -------- Background Sync
self.addEventListener("sync", (event) => {
  if (event.tag === BROADCAST_QUEUE) {
    event.waitUntil(syncBroadcasts());
  }
});

// -------- Utility: IndexedDB setup
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { autoIncrement: true });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = (e) => reject(e.target.error);
  });
}

async function queueBroadcast(request) {
  const body = await request.json();
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  tx.objectStore(STORE_NAME).add(body);
  await tx.complete;

  // Register sync
  if ("sync" in self.registration) {
    await self.registration.sync.register(BROADCAST_QUEUE);
  }
}

async function syncBroadcasts() {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);

  const getAll = store.getAll();
  getAll.onsuccess = async () => {
    const broadcasts = getAll.result;

    for (const payload of broadcasts) {
      try {
        const res = await fetch("/broadcast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          console.log("✅ Synced broadcast:", payload);
          store.delete(payload.id);
        } else {
          console.warn("⚠️ Broadcast sync failed:", res.status);
        }
      } catch (e) {
        console.error("❌ Error syncing broadcast:", e);
      }
    }
  };
}