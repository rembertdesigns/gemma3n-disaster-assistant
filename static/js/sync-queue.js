import { openDB } from '/static/js/idb.mjs';

const DB_NAME = 'report_sync_queue';
const STORE = 'queued_reports';

export async function queueReport(data) {
  const db = await openDB(DB_NAME, 1, {
    upgrade(db) {
      db.createObjectStore(STORE, { autoIncrement: true });
    },
  });
  await db.add(STORE, data);
}

export async function trySyncQueuedReports() {
  const db = await openDB(DB_NAME, 1);
  const all = await db.getAllKeys(STORE);

  for (const key of all) {
    const report = await db.get(STORE, key);
    try {
      const res = await fetch('/generate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(report),
      });

      if (res.ok) {
        await db.delete(STORE, key);
        console.log('✅ Synced queued report:', key);
      }
    } catch (err) {
      console.warn('⚠️ Sync failed for report:', key);
    }
  }
}