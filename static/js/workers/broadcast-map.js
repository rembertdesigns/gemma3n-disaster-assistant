let map = null;
let markersLayer = null;

export function initBroadcastMap() {
  map = L.map("broadcast-map").setView([34.05, -118.25], 10); // Default LA view
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  markersLayer = L.layerGroup().addTo(map);
}

export function updateBroadcastPins(broadcasts) {
  if (!map || !markersLayer) return;

  markersLayer.clearLayers();

  broadcasts.forEach((b) => {
    const { lat, lon } = b.location || {};
    if (lat && lon) {
      const marker = L.marker([lat, lon]).bindPopup(`
        <strong>🚨 ${b.severity}</strong><br/>
        ${b.message}<br/>
        <small>${b.timestamp}</small>
      `);
      markersLayer.addLayer(marker);
    }
  });
}