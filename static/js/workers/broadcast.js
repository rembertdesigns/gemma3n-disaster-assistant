import {
  initP2PFallback,
  sendP2PBroadcast,
  isP2PActive
} from './fallback-webrtc.js';
import { updateBroadcastMap } from './broadcast-map.js';

let pollInterval = null;
let peerCache = [];

export function startBroadcastListener(intervalMs = 15000) {
  fetchAndRenderBroadcasts();
  initP2PFallback(triggerLocalBroadcast);
  pollInterval = setInterval(fetchAndRenderBroadcasts, intervalMs);
  setupDiagnostics();
}

async function fetchAndRenderBroadcasts() {
  if (isP2PActive()) return;
  try {
    const res = await fetch('/broadcasts');
    const broadcasts = await res.json();
    peerCache = broadcasts;
    renderAllBroadcasts();
  } catch (err) {
    console.error('❌ Fetch error:', err);
  }
}

function renderAllBroadcasts() {
  const feed = document.getElementById('broadcast-feed');
  const alert = document.getElementById('broadcast-alert');
  const all = [...peerCache];
  if (!feed) return;
  if (all.length === 0) {
    feed.innerHTML = '<em>No active emergency broadcasts.</em>';
    if (alert) alert.style.display = 'none';
    return;
  }
  feed.innerHTML = all.map(renderBroadcastCard).join('');
  if (alert) alert.style.display = 'block';
  updateBroadcastMap(all);
}

function triggerLocalBroadcast(data) {
  const feed = document.getElementById('broadcast-feed');
  const alert = document.getElementById('broadcast-alert');
  if (!data || !feed) return;
  const newCard = renderBroadcastCard({ ...data, timestamp: `P2P · ${new Date().toISOString()}` });
  feed.innerHTML = newCard + feed.innerHTML;
  if (alert) alert.style.display = 'block';
  peerCache.unshift(data);
  updateBroadcastMap(peerCache);
}

export function submitBroadcast(payload) {
  if (isP2PActive()) {
    sendP2PBroadcast(payload);
  } else {
    fetch('/api/broadcast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }
}

function renderBroadcastCard(b) {
  const toneBadge = getToneBadge(b.tone);
  const escalationBadge = getEscalationBadge(b.escalation);
  const sentimentIcon = getSentimentIcon(b.sentiment);
  return `
    <div class="broadcast-card">
      <p><strong>${sentimentIcon} ${b.severity}</strong> ${toneBadge} ${escalationBadge}</p>
      <p>${b.message}</p>
      <small><code>${b.timestamp}</code></small>
      <hr/>
    </div>`;
}

function getToneBadge(tone) {
  const colors = {
    Descriptive: 'gray', Urgent: 'orange', Frantic: 'red', Helpless: 'purple'
  };
  return tone ? `<span style="background:${colors[tone] || '#ccc'};padding:2px 6px;border-radius:4px;color:white;font-size:0.8em">${tone}</span>` : '';
}

function getEscalationBadge(level) {
  const map = {
    Low: '🟢 Low', Moderate: '🟡 Moderate', High: '🟠 High', Critical: '🔴 Critical'
  };
  return level ? `<span style="margin-left:8px;font-weight:bold">${map[level] || level}</span>` : '';
}

function getSentimentIcon(sentiment) {
  const map = {
    Calm: '🧘', Concerned: '🤔', Anxious: '😰', Panic: '🚨'
  };
  return map[sentiment] || '📢';
}

function setupDiagnostics() {
  setInterval(() => {
    const statusEl = document.getElementById('mesh-mode');
    const peerCountEl = document.getElementById('peer-count');
    if (statusEl) statusEl.textContent = isP2PActive() ? 'Connected' : 'Polling';
    if (peerCountEl) peerCountEl.textContent = peers.length;
  }, 3000);
}