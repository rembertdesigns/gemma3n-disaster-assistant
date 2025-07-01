import { openDB } from '/static/js/idb.mjs';

const DB_NAME = 'report_sync_queue';
const STORE = 'queued_reports';
let syncInterval = null;
let statusCallback = null;

export async function queueReport(data) {
  const db = await openDB(DB_NAME, 1, {
    upgrade(db) {
      db.createObjectStore(STORE, { autoIncrement: true });
    },
  });
  await db.add(STORE, data);
  if (statusCallback) statusCallback({ syncing: false, queued: await getQueueLength() });
}

export async function getQueuedReports() {
  const db = await openDB(DB_NAME, 1);
  const keys = await db.getAllKeys(STORE);
  const reports = await db.getAll(STORE);
  return reports.map((r, i) => ({ id: keys[i], ...r }));
}

export async function removeQueuedReport(key) {
  const db = await openDB(DB_NAME, 1);
  await db.delete(STORE, key);
  if (statusCallback) statusCallback({ syncing: false, queued: await getQueueLength() });
}

export async function getQueueLength() {
  const db = await openDB(DB_NAME, 1);
  const keys = await db.getAllKeys(STORE);
  return keys.length;
}

export function setSyncStatusCallback(callback) {
  statusCallback = callback;
}

export async function trySyncQueuedReports() {
  const db = await openDB(DB_NAME, 1);
  const keys = await db.getAllKeys(STORE);

  if (keys.length === 0 && statusCallback) {
    statusCallback({ syncing: false, queued: 0 });
    return;
  }

  if (statusCallback) statusCallback({ syncing: true, queued: keys.length });

  for (const key of keys) {
    const report = await db.get(STORE, key);
    try {
      const res = await fetch('/generate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(report),
      });

      if (res.ok) {
        await db.delete(STORE, key);
        console.log('✅ Synced report:', key);
      } else {
        console.warn('⚠️ Server rejected report:', key);
      }
    } catch (err) {
      console.warn('⚠️ Sync failed for report:', key, err);
    }
  }

  const remaining = await getQueueLength();
  if (statusCallback) statusCallback({ syncing: false, queued: remaining });
}

export function startAutoRetry(intervalMs = 30000) {
  if (syncInterval) clearInterval(syncInterval);
  syncInterval = setInterval(() => {
    if (navigator.onLine) trySyncQueuedReports();
  }, intervalMs);
}

export function stopAutoRetry() {
  if (syncInterval) clearInterval(syncInterval);
  syncInterval = null;
}