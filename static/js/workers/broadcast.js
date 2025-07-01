// broadcast.js
import { updateBroadcastMap } from "./broadcast-map.js";

let pollInterval = null;

export function startBroadcastListener(intervalMs = 15000) {
  fetchAndRenderBroadcasts(); // Load immediately
  pollInterval = setInterval(fetchAndRenderBroadcasts, intervalMs);
}

export function stopBroadcastListener() {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}

async function fetchAndRenderBroadcasts() {
  try {
    const response = await fetch("/broadcasts");
    const broadcasts = await response.json();

    const container = document.getElementById("broadcast-feed");
    const alertBanner = document.getElementById("broadcast-alert");

    if (!container) return;

    if (broadcasts.length === 0) {
      container.innerHTML = "<em>No active emergency broadcasts.</em>";
      if (alertBanner) alertBanner.style.display = "none";
      return;
    }

    // Render broadcast cards
    container.innerHTML = broadcasts.map(b => renderBroadcastCard(b)).join("");

    // Show alert banner
    if (alertBanner) alertBanner.style.display = "block";

    // Update map pins
    updateBroadcastMap(broadcasts);

  } catch (error) {
    console.error("❌ Error fetching broadcasts:", error);
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
    </div>
  `;
}

function getToneBadge(tone) {
  const colors = {
    Descriptive: "gray",
    Urgent: "orange",
    Frantic: "red",
    Helpless: "purple"
  };
  return tone ? `<span style="background:${colors[tone] || '#ccc'};padding:2px 6px;border-radius:4px;color:white;font-size:0.8em">${tone}</span>` : "";
}

function getEscalationBadge(level) {
  const map = {
    Low: "🟢 Low",
    Moderate: "🟡 Moderate",
    High: "🟠 High",
    Critical: "🔴 Critical"
  };
  return level ? `<span style="margin-left:8px;font-weight:bold">${map[level] || level}</span>` : "";
}

function getSentimentIcon(sentiment) {
  const map = {
    Calm: "🧘",
    Concerned: "🤔",
    Anxious: "😰",
    Panic: "🚨"
  };
  return map[sentiment] || "📢";
}