let map = null;
let markersLayer = null;
let userLocationMarker = null;
let heatmapLayer = null;

// Severity levels with colors and icons
const SEVERITY_CONFIG = {
  'CRITICAL': { color: '#ff0000', icon: '🚨', priority: 4 },
  'HIGH': { color: '#ff6600', icon: '⚠️', priority: 3 },
  'MEDIUM': { color: '#ffcc00', icon: '⚡', priority: 2 },
  'LOW': { color: '#00cc00', icon: 'ℹ️', priority: 1 },
  'INFO': { color: '#0066cc', icon: '📢', priority: 0 }
};

// Map themes
const MAP_THEMES = {
  standard: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  terrain: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
};

let currentTheme = 'standard';
let currentTileLayer = null;

export function initBroadcastMap(options = {}) {
  const {
    containerId = "broadcast-map",
    center = [34.05, -118.25],
    zoom = 10,
    theme = 'standard',
    enableGeolocation = true,
    enableClustering = true,
    enableHeatmap = false
  } = options;

  // Initialize map
  map = L.map(containerId, {
    zoomControl: false,
    attributionControl: false
  }).setView(center, zoom);

  // Add custom zoom control
  L.control.zoom({
    position: 'topright'
  }).addTo(map);

  // Set initial theme
  setMapTheme(theme);

  // Initialize layers
  if (enableClustering) {
    markersLayer = L.markerClusterGroup({
      chunkedLoading: true,
      maxClusterRadius: 50,
      iconCreateFunction: function(cluster) {
        const markers = cluster.getAllChildMarkers();
        const severity = getHighestSeverity(markers);
        const config = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.INFO;
        
        return L.divIcon({
          html: `<div style="background-color: ${config.color}; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${cluster.getChildCount()}</div>`,
          className: 'custom-cluster-icon',
          iconSize: [30, 30]
        });
      }
    });
  } else {
    markersLayer = L.layerGroup();
  }
  
  markersLayer.addTo(map);

  // Add custom controls
  addCustomControls();

  // Get user location if enabled
  if (enableGeolocation) {
    getUserLocation();
  }

  // Add map event listeners
  map.on('zoomend', onZoomChange);
  map.on('moveend', onMapMove);

  console.log('Broadcast map initialized successfully');
}

function addCustomControls() {
  // Theme selector
  const themeControl = L.control({ position: 'topleft' });
  themeControl.onAdd = function() {
    const div = L.DomUtil.create('div', 'leaflet-control-custom theme-selector');
    div.innerHTML = `
      <select id="theme-selector" style="padding: 5px; border: none; border-radius: 3px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
        <option value="standard">Standard</option>
        <option value="satellite">Satellite</option>
        <option value="dark">Dark</option>
        <option value="terrain">Terrain</option>
      </select>
    `;
    
    div.addEventListener('change', function(e) {
      setMapTheme(e.target.value);
    });
    
    return div;
  };
  themeControl.addTo(map);

  // Legend
  const legendControl = L.control({ position: 'bottomright' });
  legendControl.onAdd = function() {
    const div = L.DomUtil.create('div', 'leaflet-control-custom legend');
    div.innerHTML = `
      <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); font-size: 12px;">
        <h4 style="margin: 0 0 8px 0; font-size: 14px;">Severity Levels</h4>
        ${Object.entries(SEVERITY_CONFIG).map(([severity, config]) => 
          `<div style="margin: 2px 0; display: flex; align-items: center;">
            <span style="width: 12px; height: 12px; background: ${config.color}; border-radius: 50%; margin-right: 5px;"></span>
            <span>${config.icon} ${severity}</span>
          </div>`
        ).join('')}
      </div>
    `;
    return div;
  };
  legendControl.addTo(map);
}

function setMapTheme(theme) {
  if (currentTileLayer) {
    map.removeLayer(currentTileLayer);
  }
  
  currentTheme = theme;
  const tileUrl = MAP_THEMES[theme] || MAP_THEMES.standard;
  
  currentTileLayer = L.tileLayer(tileUrl, {
    attribution: getAttribution(theme),
    maxZoom: 18
  }).addTo(map);
}

function getAttribution(theme) {
  switch (theme) {
    case 'satellite':
      return '&copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics';
    case 'dark':
      return '&copy; CartoDB &copy; OpenStreetMap contributors';
    case 'terrain':
      return '&copy; OpenTopoMap (CC-BY-SA)';
    default:
      return '&copy; OpenStreetMap contributors';
  }
}

function getUserLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        
        if (userLocationMarker) {
          map.removeLayer(userLocationMarker);
        }
        
        userLocationMarker = L.marker([latitude, longitude], {
          icon: L.divIcon({
            html: '<div style="background: #4285f4; width: 12px; height: 12px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 0 2px #4285f4;"></div>',
            className: 'user-location-marker',
            iconSize: [18, 18],
            iconAnchor: [9, 9]
          })
        }).addTo(map);
        
        userLocationMarker.bindPopup("📍 Your Location").openPopup();
        
        // Optionally center map on user location
        // map.setView([latitude, longitude], 12);
      },
      (error) => {
        console.warn('Geolocation error:', error.message);
      }
    );
  }
}

export function updateBroadcastPins(broadcasts, options = {}) {
  if (!map || !markersLayer) return;
  
  const {
    autoFit = false,
    showHeatmap = false,
    filterSeverity = null,
    maxAge = null // in minutes
  } = options;

  // Clear existing markers
  markersLayer.clearLayers();
  
  if (showHeatmap && heatmapLayer) {
    map.removeLayer(heatmapLayer);
    heatmapLayer = null;
  }

  // Filter broadcasts
  let filteredBroadcasts = broadcasts.filter(b => {
    // Filter by severity
    if (filterSeverity && b.severity !== filterSeverity) return false;
    
    // Filter by age
    if (maxAge && b.timestamp) {
      const broadcastTime = new Date(b.timestamp);
      const now = new Date();
      const ageMinutes = (now - broadcastTime) / (1000 * 60);
      if (ageMinutes > maxAge) return false;
    }
    
    return true;
  });

  // Sort by severity (highest first)
  filteredBroadcasts.sort((a, b) => {
    const priorityA = SEVERITY_CONFIG[a.severity]?.priority || 0;
    const priorityB = SEVERITY_CONFIG[b.severity]?.priority || 0;
    return priorityB - priorityA;
  });

  const markers = [];
  const heatmapData = [];

  filteredBroadcasts.forEach((broadcast, index) => {
    const { lat, lon } = broadcast.location || {};
    if (!lat || !lon) return;

    const config = SEVERITY_CONFIG[broadcast.severity] || SEVERITY_CONFIG.INFO;
    const timeAgo = broadcast.timestamp ? getTimeAgo(broadcast.timestamp) : 'Unknown time';
    
    // Create custom marker
    const marker = L.marker([lat, lon], {
      icon: L.divIcon({
        html: `<div style="background: ${config.color}; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3); animation: pulse 2s infinite;">${config.icon}</div>`,
        className: 'custom-broadcast-marker',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      }),
      zIndexOffset: config.priority * 100 // Higher severity markers appear on top
    });

    // Enhanced popup content
    const popupContent = `
      <div style="min-width: 200px; font-family: Arial, sans-serif;">
        <div style="background: ${config.color}; color: white; padding: 8px; margin: -10px -10px 10px -10px; border-radius: 3px 3px 0 0;">
          <strong>${config.icon} ${broadcast.severity}</strong>
        </div>
        <div style="margin-bottom: 8px;">
          <strong>Message:</strong><br/>
          ${broadcast.message || 'No message provided'}
        </div>
        <div style="margin-bottom: 8px;">
          <strong>Time:</strong> ${timeAgo}
        </div>
        ${broadcast.description ? `<div style="margin-bottom: 8px;"><strong>Details:</strong><br/>${broadcast.description}</div>` : ''}
        ${broadcast.contact ? `<div style="margin-bottom: 8px;"><strong>Contact:</strong> ${broadcast.contact}</div>` : ''}
        <div style="color: #666; font-size: 11px;">
          📍 ${lat.toFixed(4)}, ${lon.toFixed(4)}
        </div>
      </div>
    `;

    marker.bindPopup(popupContent, {
      maxWidth: 300,
      className: 'custom-popup'
    });

    // Add click event for additional actions
    marker.on('click', function() {
      onMarkerClick(broadcast, marker);
    });

    markersLayer.addLayer(marker);
    markers.push(marker);
    
    // Collect heatmap data
    if (showHeatmap) {
      heatmapData.push([lat, lon, config.priority + 1]);
    }
  });

  // Add heatmap if requested
  if (showHeatmap && heatmapData.length > 0) {
    // Note: This requires leaflet-heatmap plugin
    // heatmapLayer = L.heatLayer(heatmapData, {radius: 25}).addTo(map);
  }

  // Auto-fit map to show all markers
  if (autoFit && markers.length > 0) {
    const group = new L.featureGroup(markers);
    map.fitBounds(group.getBounds().pad(0.1));
  }

  console.log(`Updated ${markers.length} broadcast markers`);
}

function getHighestSeverity(markers) {
  let highest = 'INFO';
  let highestPriority = -1;
  
  markers.forEach(marker => {
    const severity = marker.options.severity || 'INFO';
    const priority = SEVERITY_CONFIG[severity]?.priority || 0;
    if (priority > highestPriority) {
      highest = severity;
      highestPriority = priority;
    }
  });
  
  return highest;
}

function getTimeAgo(timestamp) {
  const now = new Date();
  const time = new Date(timestamp);
  const diffMs = now - time;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  
  return time.toLocaleDateString();
}

function onMarkerClick(broadcast, marker) {
  // Emit custom event for external handlers
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('broadcastMarkerClick', {
      detail: { broadcast, marker }
    }));
  }
}

function onZoomChange() {
  // Adjust marker sizes based on zoom level
  const zoom = map.getZoom();
  const size = Math.max(16, Math.min(32, zoom * 2));
  
  // Update marker sizes if needed
  // This would require rebuilding markers with new sizes
}

function onMapMove() {
  // Emit map move event for external handlers
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('broadcastMapMove', {
      detail: { 
        center: map.getCenter(),
        zoom: map.getZoom(),
        bounds: map.getBounds()
      }
    }));
  }
}

// Utility functions
export function focusOnBroadcast(broadcastId, broadcasts) {
  const broadcast = broadcasts.find(b => b.id === broadcastId);
  if (broadcast && broadcast.location) {
    const { lat, lon } = broadcast.location;
    map.setView([lat, lon], 15);
    
    // Find and open the marker popup
    markersLayer.eachLayer(layer => {
      if (layer.getLatLng().lat === lat && layer.getLatLng().lng === lon) {
        layer.openPopup();
      }
    });
  }
}

export function filterBroadcastsBySeverity(severity) {
  // This would trigger a re-render with filtered data
  console.log(`Filtering broadcasts by severity: ${severity}`);
}

export function exportMapData() {
  const bounds = map.getBounds();
  const center = map.getCenter();
  const zoom = map.getZoom();
  
  return {
    bounds: {
      north: bounds.getNorth(),
      south: bounds.getSouth(),
      east: bounds.getEast(),
      west: bounds.getWest()
    },
    center: {
      lat: center.lat,
      lng: center.lng
    },
    zoom: zoom,
    theme: currentTheme
  };
}

export function clearAllMarkers() {
  if (markersLayer) {
    markersLayer.clearLayers();
  }
  if (heatmapLayer) {
    map.removeLayer(heatmapLayer);
    heatmapLayer = null;
  }
}

export function getMapInstance() {
  return map;
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
  }
  
  .custom-broadcast-marker {
    animation: pulse 2s infinite;
  }
  
  .custom-popup .leaflet-popup-content-wrapper {
    border-radius: 8px;
  }
  
  .theme-selector select {
    font-size: 12px;
    cursor: pointer;
  }
  
  .legend {
    font-family: Arial, sans-serif;
  }
`;
document.head.appendChild(style);