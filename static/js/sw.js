const CACHE_NAME = "disaster-assistant-cache-v1";
const OFFLINE_URL = "/offline.html";
const BROADCAST_QUEUE = "broadcast-sync-queue";
const REPORT_QUEUE = "crowd-report-sync-queue";
const DB_NAME = "DisasterBroadcastsDB";
const STORE_NAME = "broadcastQueue";
const REPORT_STORE = "crowdReportQueue";

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
      Promise.all(keys.map((key) => key !== CACHE_NAME && caches.delete(key)))
    )
  );
  self.clients.claim();
});

// -------- Fetch Handler
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const urlPath = new URL(request.url).pathname;

  // POST: /broadcast
  if (request.method === "POST" && urlPath === "/broadcast") {
    event.respondWith(
      fetch(request.clone()).catch(() =>
        queueBroadcast(request.clone()).then(() =>
          new Response(JSON.stringify({ status: "queued", offline: true }), {
            headers: { "Content-Type": "application/json" },
          })
        )
      )
    );
    return;
  }

  // POST: /api/submit-crowd-report
  if (request.method === "POST" && urlPath === "/api/submit-crowd-report") {
    event.respondWith(
      fetch(request.clone()).catch(() =>
        queueCrowdReport(request.clone()).then(() =>
          new Response(JSON.stringify({ status: "queued", offline: true }), {
            headers: { "Content-Type": "application/json" },
          })
        )
      )
    );
    return;
  }

  // GET fallback
  if (request.method === "GET") {
    event.respondWith(
      caches.match(request).then((res) =>
        res ||
        fetch(request).catch(() =>
          caches.match(OFFLINE_URL)
        )
      )
    );
  }
});

// -------- Background Sync Events
self.addEventListener("sync", (event) => {
  if (event.tag === BROADCAST_QUEUE) {
    event.waitUntil(syncBroadcasts());
  }
  if (event.tag === REPORT_QUEUE) {
    event.waitUntil(syncCrowdReports());
  }
});

// -------- IndexedDB Setup
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 2);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { autoIncrement: true });
      }
      if (!db.objectStoreNames.contains(REPORT_STORE)) {
        db.createObjectStore(REPORT_STORE, { autoIncrement: true });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = (e) => reject(e.target.error);
  });
}

// -------- Broadcast Queue Logic
async function queueBroadcast(request) {
  const body = await request.json();
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  tx.objectStore(STORE_NAME).add(body);
  await tx.done;

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
    const broadcasts = getAll.result || [];
    console.log(`📡 Syncing ${broadcasts.length} broadcast(s)...`);

    for (const [index, payload] of broadcasts.entries()) {
      try {
        const res = await fetch("/broadcast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          store.delete(index + 1);
          console.log("✅ Broadcast synced:", payload);
        } else {
          console.warn("⚠️ Broadcast sync failed:", res.status);
        }
      } catch (e) {
        console.error("❌ Broadcast error:", e);
      }
    }
  };

  getAll.onerror = (e) => console.error("🛑 Error reading broadcast queue", e);
}

// -------- Crowd Report Queue Logic
async function queueCrowdReport(request) {
  const formData = await request.formData();
  const data = {};
  formData.forEach((value, key) => {
    data[key] = value;
  });

  const db = await openDB();
  const tx = db.transaction(REPORT_STORE, "readwrite");
  tx.objectStore(REPORT_STORE).add(data);
  await tx.done;

  if ("sync" in self.registration) {
    await self.registration.sync.register(REPORT_QUEUE);
  }
}

async function syncCrowdReports() {
  const db = await openDB();
  const tx = db.transaction(REPORT_STORE, "readwrite");
  const store = tx.objectStore(REPORT_STORE);
  const getAll = store.getAll();

  getAll.onsuccess = async () => {
    const reports = getAll.result || [];
    console.log(`📝 Syncing ${reports.length} crowd report(s)...`);

    for (const [index, report] of reports.entries()) {
      try {
        const formBody = new FormData();
        Object.entries(report).forEach(([key, val]) =>
          formBody.append(key, val)
        );

        const res = await fetch("/api/submit-crowd-report", {
          method: "POST",
          body: formBody,
        });

        if (res.ok) {
          store.delete(index + 1);
          console.log("✅ Report synced:", report);
        } else {
          console.warn("⚠️ Crowd report sync failed:", res.status);
        }
      } catch (e) {
        console.error("❌ Report sync error:", e);
      }
    }
  };

  getAll.onerror = (e) => console.error("🛑 Error reading crowd report queue", e);
}