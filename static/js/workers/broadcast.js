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
  return `
    <div class="broadcast-card">
      <p><strong>🚨 ${b.severity}</strong></p>
      <p>${b.message}</p>
      <small><code>${b.timestamp}</code></small>
      <hr/>
    </div>
  `;
}