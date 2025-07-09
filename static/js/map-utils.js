/**
 * 🗺️ Map Utilities - Emergency Response Mapping System
 * 
 * Comprehensive mapping utilities for emergency response applications
 * Includes location services, geocoding, routing, and emergency overlays
 */

// Configuration
const MAP_CONFIG = {
    DEFAULT_CENTER: [34.05, -118.25], // Los Angeles default
    DEFAULT_ZOOM: 10,
    MAX_ZOOM: 18,
    MIN_ZOOM: 2,
    
    // Map tile providers
    TILE_PROVIDERS: {
      osm: {
        url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
      },
      satellite: {
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attribution: '&copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
        maxZoom: 18
      },
      terrain: {
        url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attribution: '&copy; OpenTopoMap (CC-BY-SA)',
        maxZoom: 17
      },
      dark: {
        url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attribution: '&copy; CartoDB &copy; OpenStreetMap contributors',
        maxZoom: 19
      }
    },
    
    // Emergency icons and colors
    EMERGENCY_MARKERS: {
      fire: { icon: '🔥', color: '#dc2626', priority: 1 },
      medical: { icon: '🚑', color: '#059669', priority: 2 },
      police: { icon: '🚔', color: '#2563eb', priority: 2 },
      hazmat: { icon: '☢️', color: '#7c2d12', priority: 1 },
      flood: { icon: '🌊', color: '#0891b2', priority: 3 },
      earthquake: { icon: '📳', color: '#92400e', priority: 1 },
      traffic: { icon: '🚧', color: '#f59e0b', priority: 4 },
      weather: { icon: '⛈️', color: '#6366f1', priority: 3 },
      general: { icon: '🚨', color: '#ef4444', priority: 5 }
    },
    
    // Geolocation settings
    GEOLOCATION_OPTIONS: {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 300000 // 5 minutes
    },
    
    // Search and routing
    GEOCODING_PROVIDER: 'nominatim',
    ROUTING_PROVIDER: 'osrm',
    
    // Performance settings
    MARKER_CLUSTERING_THRESHOLD: 50,
    UPDATE_THROTTLE: 1000, // 1 second
    MAX_MARKERS: 1000
  };
  
  // Global state
  let mapInstances = new Map();
  let userLocation = null;
  let locationWatchId = null;
  let geocodingCache = new Map();
  let routingCache = new Map();
  
  /**
   * Initialize a new map instance
   */
  export function initializeMap(containerId, options = {}) {
    try {
      const config = {
        center: options.center || MAP_CONFIG.DEFAULT_CENTER,
        zoom: options.zoom || MAP_CONFIG.DEFAULT_ZOOM,
        theme: options.theme || 'osm',
        enableGeolocation: options.enableGeolocation !== false,
        enableClustering: options.enableClustering !== false,
        enableRouting: options.enableRouting !== false,
        enableEmergencyLayers: options.enableEmergencyLayers !== false,
        ...options
      };
      
      // Create map instance
      const map = L.map(containerId, {
        zoomControl: false,
        attributionControl: true,
        preferCanvas: true, // Better performance
        maxZoom: MAP_CONFIG.MAX_ZOOM,
        minZoom: MAP_CONFIG.MIN_ZOOM
      }).setView(config.center, config.zoom);
      
      // Add tile layer
      setMapTheme(map, config.theme);
      
      // Add custom controls
      addMapControls(map, config);
      
      // Initialize layers
      const layers = initializeMapLayers(map, config);
      
      // Setup event handlers
      setupMapEvents(map, config);
      
      // Enable geolocation if requested
      if (config.enableGeolocation) {
        enableGeolocation(map);
      }
      
      // Store map instance
      const mapData = {
        instance: map,
        layers,
        config,
        markers: new Map(),
        routes: new Map(),
        overlays: new Map()
      };
      
      mapInstances.set(containerId, mapData);
      
      console.log(`🗺️ Map initialized: ${containerId}`);
      return map;
      
    } catch (error) {
      console.error('❌ Failed to initialize map:', error);
      throw error;
    }
  }
  
  /**
   * Set map theme/tile provider
   */
  export function setMapTheme(map, theme) {
    // Remove existing tile layers
    map.eachLayer(layer => {
      if (layer instanceof L.TileLayer) {
        map.removeLayer(layer);
      }
    });
    
    const provider = MAP_CONFIG.TILE_PROVIDERS[theme] || MAP_CONFIG.TILE_PROVIDERS.osm;
    
    const tileLayer = L.tileLayer(provider.url, {
      attribution: provider.attribution,
      maxZoom: provider.maxZoom
    });
    
    tileLayer.addTo(map);
    
    // Update map instance data
    for (const [containerId, mapData] of mapInstances) {
      if (mapData.instance === map) {
        mapData.config.theme = theme;
        break;
      }
    }
  }
  
  /**
   * Add emergency marker to map
   */
  export function addEmergencyMarker(mapOrContainerId, incident) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return null;
      
      const { instance: map, markers } = mapData;
      
      // Validate incident data
      if (!incident.coordinates || incident.coordinates.length < 2) {
        throw new Error('Invalid coordinates for emergency marker');
      }
      
      const [lat, lng] = incident.coordinates;
      const emergencyType = incident.type || 'general';
      const markerConfig = MAP_CONFIG.EMERGENCY_MARKERS[emergencyType] || MAP_CONFIG.EMERGENCY_MARKERS.general;
      
      // Create custom marker icon
      const markerIcon = L.divIcon({
        className: 'emergency-marker',
        html: `
          <div class="emergency-marker-container" style="
            background-color: ${markerConfig.color};
            color: white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            animation: emergency-pulse 2s infinite;
          ">
            ${markerConfig.icon}
          </div>
        `,
        iconSize: [40, 40],
        iconAnchor: [20, 20]
      });
      
      // Create marker with popup
      const marker = L.marker([lat, lng], { icon: markerIcon });
      
      // Create popup content
      const popupContent = createEmergencyPopup(incident, markerConfig);
      marker.bindPopup(popupContent, {
        maxWidth: 300,
        className: 'emergency-popup'
      });
      
      // Add to map and store reference
      marker.addTo(map);
      markers.set(incident.id, marker);
      
      // Add to clustering layer if enabled
      if (mapData.layers.markerCluster) {
        mapData.layers.markerCluster.addLayer(marker);
      }
      
      console.log(`🚨 Emergency marker added: ${incident.id} (${emergencyType})`);
      return marker;
      
    } catch (error) {
      console.error('❌ Failed to add emergency marker:', error);
      return null;
    }
  }
  
  /**
   * Remove emergency marker from map
   */
  export function removeEmergencyMarker(mapOrContainerId, incidentId) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return false;
      
      const { instance: map, markers, layers } = mapData;
      const marker = markers.get(incidentId);
      
      if (marker) {
        map.removeLayer(marker);
        if (layers.markerCluster) {
          layers.markerCluster.removeLayer(marker);
        }
        markers.delete(incidentId);
        console.log(`🗑️ Emergency marker removed: ${incidentId}`);
        return true;
      }
      
      return false;
      
    } catch (error) {
      console.error('❌ Failed to remove emergency marker:', error);
      return false;
    }
  }
  
  /**
   * Update emergency marker
   */
  export function updateEmergencyMarker(mapOrContainerId, incident) {
    removeEmergencyMarker(mapOrContainerId, incident.id);
    return addEmergencyMarker(mapOrContainerId, incident);
  }
  
  /**
   * Get all emergency markers
   */
  export function getAllEmergencyMarkers(mapOrContainerId) {
    const mapData = getMapData(mapOrContainerId);
    return mapData ? Array.from(mapData.markers.values()) : [];
  }
  
  /**
   * Clear all emergency markers
   */
  export function clearAllEmergencyMarkers(mapOrContainerId) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return false;
      
      const { instance: map, markers, layers } = mapData;
      
      for (const marker of markers.values()) {
        map.removeLayer(marker);
        if (layers.markerCluster) {
          layers.markerCluster.removeLayer(marker);
        }
      }
      
      markers.clear();
      console.log('🧹 All emergency markers cleared');
      return true;
      
    } catch (error) {
      console.error('❌ Failed to clear emergency markers:', error);
      return false;
    }
  }
  
  /**
   * Add custom controls to map
   */
  function addMapControls(map, config) {
    // Custom zoom control
    const zoomControl = L.control.zoom({
      position: 'topright'
    });
    map.addControl(zoomControl);
    
    // Theme switcher control
    const themeControl = L.control({
      position: 'topright'
    });
    
    themeControl.onAdd = function() {
      const div = L.DomUtil.create('div', 'leaflet-control-theme');
      div.innerHTML = `
        <select id="theme-selector" class="theme-selector">
          <option value="osm">Street</option>
          <option value="satellite">Satellite</option>
          <option value="terrain">Terrain</option>
          <option value="dark">Dark</option>
        </select>
      `;
      
      const select = div.querySelector('#theme-selector');
      select.value = config.theme;
      
      L.DomEvent.on(select, 'change', function(e) {
        setMapTheme(map, e.target.value);
      });
      
      L.DomEvent.disableClickPropagation(div);
      return div;
    };
    
    map.addControl(themeControl);
    
    // Geolocation control
    if (config.enableGeolocation) {
      const locationControl = L.control({
        position: 'topright'
      });
      
      locationControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'leaflet-control-location');
        div.innerHTML = `
          <button id="location-btn" class="location-btn" title="Find my location">
            📍
          </button>
        `;
        
        const button = div.querySelector('#location-btn');
        L.DomEvent.on(button, 'click', function() {
          enableGeolocation(map);
        });
        
        L.DomEvent.disableClickPropagation(div);
        return div;
      };
      
      map.addControl(locationControl);
    }
    
    // Emergency layers control
    if (config.enableEmergencyLayers) {
      const layerControl = L.control({
        position: 'bottomright'
      });
      
      layerControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'leaflet-control-emergency');
        div.innerHTML = `
          <div class="emergency-layers-panel">
            <h4>Emergency Layers</h4>
            <label><input type="checkbox" id="fire-layer" checked> Fire 🔥</label>
            <label><input type="checkbox" id="medical-layer" checked> Medical 🚑</label>
            <label><input type="checkbox" id="police-layer" checked> Police 🚔</label>
            <label><input type="checkbox" id="hazmat-layer" checked> Hazmat ☢️</label>
            <label><input type="checkbox" id="flood-layer" checked> Flood 🌊</label>
            <label><input type="checkbox" id="earthquake-layer" checked> Earthquake 📳</label>
            <label><input type="checkbox" id="traffic-layer" checked> Traffic 🚧</label>
            <label><input type="checkbox" id="weather-layer" checked> Weather ⛈️</label>
          </div>
        `;
        
        // Add event listeners for layer toggles
        const checkboxes = div.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
          L.DomEvent.on(checkbox, 'change', function(e) {
            const layerType = e.target.id.replace('-layer', '');
            toggleEmergencyLayer(map, layerType, e.target.checked);
          });
        });
        
        L.DomEvent.disableClickPropagation(div);
        return div;
      };
      
      map.addControl(layerControl);
    }
  }
  
  /**
   * Initialize map layers
   */
  function initializeMapLayers(map, config) {
    const layers = {};
    
    // Marker clustering layer
    if (config.enableClustering) {
      layers.markerCluster = L.markerClusterGroup({
        maxClusterRadius: 50,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        zoomToBoundsOnClick: true
      });
      map.addLayer(layers.markerCluster);
    }
    
    // Emergency type layers
    if (config.enableEmergencyLayers) {
      Object.keys(MAP_CONFIG.EMERGENCY_MARKERS).forEach(type => {
        layers[type] = L.layerGroup();
        map.addLayer(layers[type]);
      });
    }
    
    // Drawing layer for routes and areas
    layers.drawing = L.layerGroup();
    map.addLayer(layers.drawing);
    
    return layers;
  }
  
  /**
   * Setup map event handlers
   */
  function setupMapEvents(map, config) {
    // Map click event
    map.on('click', function(e) {
      console.log(`📍 Map clicked at: ${e.latlng.lat}, ${e.latlng.lng}`);
      
      // Emit custom event
      map.fire('mapClick', {
        latlng: e.latlng,
        coordinates: [e.latlng.lat, e.latlng.lng]
      });
    });
    
    // Map zoom event
    map.on('zoomend', function() {
      const zoom = map.getZoom();
      console.log(`🔍 Map zoom changed to: ${zoom}`);
      
      // Emit custom event
      map.fire('mapZoomChange', { zoom });
    });
    
    // Map move event (throttled)
    let moveTimeout;
    map.on('moveend', function() {
      clearTimeout(moveTimeout);
      moveTimeout = setTimeout(() => {
        const center = map.getCenter();
        console.log(`📍 Map moved to: ${center.lat}, ${center.lng}`);
        
        // Emit custom event
        map.fire('mapMoveEnd', {
          center: center,
          coordinates: [center.lat, center.lng]
        });
      }, MAP_CONFIG.UPDATE_THROTTLE);
    });
  }
  
  /**
   * Enable geolocation
   */
  function enableGeolocation(map) {
    if (!navigator.geolocation) {
      console.warn('⚠️ Geolocation not supported');
      return;
    }
    
    // Stop watching if already active
    if (locationWatchId) {
      navigator.geolocation.clearWatch(locationWatchId);
    }
    
    // Get current position
    navigator.geolocation.getCurrentPosition(
      position => {
        const { latitude, longitude } = position.coords;
        userLocation = [latitude, longitude];
        
        // Center map on user location
        map.setView(userLocation, 15);
        
        // Add/update user location marker
        updateUserLocationMarker(map, position);
        
        console.log(`📍 User location: ${latitude}, ${longitude}`);
        
        // Emit custom event
        map.fire('userLocationFound', {
          coordinates: userLocation,
          accuracy: position.coords.accuracy
        });
        
        // Start watching position
        locationWatchId = navigator.geolocation.watchPosition(
          position => {
            const { latitude, longitude } = position.coords;
            userLocation = [latitude, longitude];
            updateUserLocationMarker(map, position);
            
            map.fire('userLocationUpdate', {
              coordinates: userLocation,
              accuracy: position.coords.accuracy
            });
          },
          error => {
            console.error('❌ Geolocation error:', error);
            map.fire('userLocationError', { error });
          },
          MAP_CONFIG.GEOLOCATION_OPTIONS
        );
      },
      error => {
        console.error('❌ Failed to get user location:', error);
        map.fire('userLocationError', { error });
      },
      MAP_CONFIG.GEOLOCATION_OPTIONS
    );
  }
  
  /**
   * Update user location marker
   */
  function updateUserLocationMarker(map, position) {
    const { latitude, longitude, accuracy } = position.coords;
    const latlng = [latitude, longitude];
    
    // Remove existing user location marker
    map.eachLayer(layer => {
      if (layer.options && layer.options.isUserLocation) {
        map.removeLayer(layer);
      }
    });
    
    // Create accuracy circle
    const accuracyCircle = L.circle(latlng, {
      radius: accuracy,
      color: '#007bff',
      fillColor: '#007bff',
      fillOpacity: 0.1,
      weight: 2,
      isUserLocation: true
    });
    
    // Create user location marker
    const userMarker = L.marker(latlng, {
      icon: L.divIcon({
        className: 'user-location-marker',
        html: `
          <div style="
            background-color: #007bff;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          ">
            📍
          </div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
      }),
      isUserLocation: true
    });
    
    // Add to map
    accuracyCircle.addTo(map);
    userMarker.addTo(map);
    
    // Add popup
    userMarker.bindPopup(`
      <div class="user-location-popup">
        <h4>📍 Your Location</h4>
        <p><strong>Coordinates:</strong> ${latitude.toFixed(6)}, ${longitude.toFixed(6)}</p>
        <p><strong>Accuracy:</strong> ±${accuracy.toFixed(0)}m</p>
        <p><strong>Timestamp:</strong> ${new Date().toLocaleTimeString()}</p>
      </div>
    `);
  }
  
  /**
   * Create emergency popup content
   */
  function createEmergencyPopup(incident, markerConfig) {
    const statusClass = incident.status === 'active' ? 'status-active' : 
                       incident.status === 'resolved' ? 'status-resolved' : 'status-pending';
    
    return `
      <div class="emergency-popup-content">
        <div class="emergency-header">
          <span class="emergency-icon">${markerConfig.icon}</span>
          <h3>${incident.title || 'Emergency Incident'}</h3>
          <span class="emergency-status ${statusClass}">${incident.status || 'pending'}</span>
        </div>
        
        <div class="emergency-details">
          <p><strong>Type:</strong> ${incident.type || 'general'}</p>
          <p><strong>Priority:</strong> ${'★'.repeat(markerConfig.priority)}</p>
          <p><strong>Location:</strong> ${incident.address || 'Unknown'}</p>
          <p><strong>Reported:</strong> ${incident.timestamp ? new Date(incident.timestamp).toLocaleString() : 'Unknown'}</p>
          ${incident.description ? `<p><strong>Description:</strong> ${incident.description}</p>` : ''}
          ${incident.reporter ? `<p><strong>Reporter:</strong> ${incident.reporter}</p>` : ''}
          ${incident.responders ? `<p><strong>Responders:</strong> ${incident.responders.join(', ')}</p>` : ''}
        </div>
        
        <div class="emergency-actions">
          <button onclick="getDirections([${incident.coordinates}])">🧭 Directions</button>
          <button onclick="updateIncident('${incident.id}')">✏️ Update</button>
          <button onclick="shareIncident('${incident.id}')">📤 Share</button>
        </div>
      </div>
    `;
  }
  
  /**
   * Toggle emergency layer visibility
   */
  function toggleEmergencyLayer(map, layerType, visible) {
    const mapData = getMapDataByMap(map);
    if (!mapData || !mapData.layers[layerType]) return;
    
    const layer = mapData.layers[layerType];
    
    if (visible) {
      if (!map.hasLayer(layer)) {
        map.addLayer(layer);
      }
    } else {
      if (map.hasLayer(layer)) {
        map.removeLayer(layer);
      }
    }
    
    console.log(`👁️ Emergency layer ${layerType}: ${visible ? 'shown' : 'hidden'}`);
  }
  
  /**
   * Geocode address to coordinates
   */
  export async function geocodeAddress(address) {
    try {
      // Check cache first
      if (geocodingCache.has(address)) {
        return geocodingCache.get(address);
      }
      
      // Use Nominatim for geocoding
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}&limit=1`
      );
      
      const data = await response.json();
      
      if (data && data.length > 0) {
        const result = {
          coordinates: [parseFloat(data[0].lat), parseFloat(data[0].lon)],
          address: data[0].display_name,
          confidence: data[0].importance || 0.5
        };
        
        // Cache result
        geocodingCache.set(address, result);
        
        return result;
      }
      
      throw new Error('Address not found');
      
    } catch (error) {
      console.error('❌ Geocoding error:', error);
      throw error;
    }
  }
  
  /**
   * Reverse geocode coordinates to address
   */
  export async function reverseGeocode(coordinates) {
    try {
      const [lat, lng] = coordinates;
      const cacheKey = `${lat},${lng}`;
      
      // Check cache first
      if (geocodingCache.has(cacheKey)) {
        return geocodingCache.get(cacheKey);
      }
      
      // Use Nominatim for reverse geocoding
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`
      );
      
      const data = await response.json();
      
      if (data && data.display_name) {
        const result = {
          address: data.display_name,
          coordinates: [lat, lng],
          components: data.address || {}
        };
        
        // Cache result
        geocodingCache.set(cacheKey, result);
        
        return result;
      }
      
      throw new Error('Address not found');
      
    } catch (error) {
      console.error('❌ Reverse geocoding error:', error);
      throw error;
    }
  }
  
  /**
   * Calculate route between points
   */
  export async function calculateRoute(start, end, options = {}) {
    try {
      const routeKey = `${start.join(',')}-${end.join(',')}`;
      
      // Check cache first
      if (routingCache.has(routeKey)) {
        return routingCache.get(routeKey);
      }
      
      // Use OSRM for routing
      const profile = options.profile || 'driving';
      const response = await fetch(
        `https://router.project-osrm.org/route/v1/${profile}/${start[1]},${start[0]};${end[1]},${end[0]}?overview=full&geometries=geojson&steps=true`
      );
      
      const data = await response.json();
      
      if (data.routes && data.routes.length > 0) {
        const route = data.routes[0];
        
        const result = {
          coordinates: route.geometry.coordinates.map(coord => [coord[1], coord[0]]),
          distance: route.distance,
          duration: route.duration,
          steps: route.legs[0].steps.map(step => ({
            instruction: step.maneuver.instruction || 'Continue',
            distance: step.distance,
            duration: step.duration
          }))
        };
        
        // Cache result
        routingCache.set(routeKey, result);
        
        return result;
      }
      
      throw new Error('Route not found');
      
    } catch (error) {
      console.error('❌ Routing error:', error);
      throw error;
    }
  }
  
  /**
   * Add route to map
   */
  export async function addRoute(mapOrContainerId, routeId, start, end, options = {}) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return null;
      
      const { instance: map, routes, layers } = mapData;
      
      // Calculate route
      const route = await calculateRoute(start, end, options);
      
      // Create route polyline
      const routeLine = L.polyline(route.coordinates, {
        color: options.color || '#007bff',
        weight: options.weight || 5,
        opacity: options.opacity || 0.8,
        dashArray: options.dashArray || null
      });
      
      // Add to map
      routeLine.addTo(map);
      layers.drawing.addLayer(routeLine);
      
      // Create route popup
      const popupContent = `
        <div class="route-popup">
          <h4>🗺️ Route Information</h4>
          <p><strong>Distance:</strong> ${(route.distance / 1000).toFixed(2)} km</p>
          <p><strong>Duration:</strong> ${Math.round(route.duration / 60)} minutes</p>
          <p><strong>Steps:</strong> ${route.steps.length}</p>
        </div>
      `;
      
      routeLine.bindPopup(popupContent);
      
      // Store route reference
      routes.set(routeId, {
        polyline: routeLine,
        data: route,
        options
      });
      
      console.log(`🗺️ Route added: ${routeId}`);
      return routeLine;
      
    } catch (error) {
      console.error('❌ Failed to add route:', error);
      return null;
    }
  }
  
  /**
   * Remove route from map
   */
  export function removeRoute(mapOrContainerId, routeId) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return false;
      
      const { instance: map, routes, layers } = mapData;
      const route = routes.get(routeId);
      
      if (route) {
        map.removeLayer(route.polyline);
        layers.drawing.removeLayer(route.polyline);
        routes.delete(routeId);
        console.log(`🗑️ Route removed: ${routeId}`);
        return true;
      }
      
      return false;
      
    } catch (error) {
      console.error('❌ Failed to remove route:', error);
      return false;
    }
  }
  
  /**
   * Get map data by container ID or map instance
   */
  function getMapData(mapOrContainerId) {
    if (typeof mapOrContainerId === 'string') {
      return mapInstances.get(mapOrContainerId);
    } else {
      return getMapDataByMap(mapOrContainerId);
    }
  }
  
  /**
   * Get map data by map instance
   */
  function getMapDataByMap(map) {
    for (const mapData of mapInstances.values()) {
      if (mapData.instance === map) {
        return mapData;
      }
    }
    return null;
  }
  
  /**
   * Cleanup map instance
   */
  export function destroyMap(containerId) {
    try {
      const mapData = mapInstances.get(containerId);
      if (mapData) {
        mapData.instance.remove();
        mapInstances.delete(containerId);
        console.log(`🗑️ Map destroyed: ${containerId}`);
        return true;
      }
      return false;
    } catch (error) {
      console.error('❌ Failed to destroy map:', error);
      return false;
    }
  }
  
  /**
   * Get user's current location
   */
  export function getUserLocation() {
    return userLocation;
  }
  
  /**
   * Set map bounds to fit all markers
   */
  export function fitMapToBounds(mapOrContainerId, padding = 20) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return false;
      
      const { instance: map, markers } = mapData;
      
      if (markers.size === 0) return false;
      
      const group = new L.featureGroup(Array.from(markers.values()));
      map.fitBounds(group.getBounds(), { padding: [padding, padding] });
      
      return true;
    } catch (error) {
      console.error('❌ Failed to fit map bounds:', error);
      return false;
    }
  }
  
  /**
   * Search for nearby emergency incidents
   */
  export function findNearbyIncidents(mapOrContainerId, coordinates, radius = 5000) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return [];
      
      const { markers } = mapData;
      const [centerLat, centerLng] = coordinates;
      const nearbyIncidents = [];
      
      for (const [incidentId, marker] of markers) {
        const markerLatLng = marker.getLatLng();
        const distance = calculateDistance(
          [centerLat, centerLng],
          [markerLatLng.lat, markerLatLng.lng]
        );
        
        if (distance <= radius) {
          nearbyIncidents.push({
            id: incidentId,
            marker,
            distance,
            coordinates: [markerLatLng.lat, markerLatLng.lng]
          });
        }
      }
      
      // Sort by distance
      nearbyIncidents.sort((a, b) => a.distance - b.distance);
      
      return nearbyIncidents;
    } catch (error) {
      console.error('❌ Failed to find nearby incidents:', error);
      return [];
    }
  }
  
  /**
   * Calculate distance between two coordinates (Haversine formula)
   */
  function calculateDistance(coord1, coord2) {
    const [lat1, lng1] = coord1;
    const [lat2, lng2] = coord2;
    
    const R = 6371e3; // Earth's radius in meters
    const φ1 = lat1 * Math.PI/180;
    const φ2 = lat2 * Math.PI/180;
    const Δφ = (lat2-lat1) * Math.PI/180;
    const Δλ = (lng2-lng1) * Math.PI/180;
    
    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
            Math.cos(φ1) * Math.cos(φ2) *
            Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    
    return R * c; // Distance in meters
  }
  
  /**
   * Add area/zone overlay to map
   */
  export function addZoneOverlay(mapOrContainerId, zoneId, coordinates, options = {}) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return null;
      
      const { instance: map, overlays, layers } = mapData;
      
      // Create polygon or circle based on coordinates
      let overlay;
      
      if (options.type === 'circle' && coordinates.length === 2) {
        // Circle zone
        overlay = L.circle(coordinates, {
          radius: options.radius || 1000,
          color: options.color || '#ff0000',
          fillColor: options.fillColor || options.color || '#ff0000',
          fillOpacity: options.fillOpacity || 0.2,
          weight: options.weight || 2,
          opacity: options.opacity || 0.8
        });
      } else {
        // Polygon zone
        overlay = L.polygon(coordinates, {
          color: options.color || '#ff0000',
          fillColor: options.fillColor || options.color || '#ff0000',
          fillOpacity: options.fillOpacity || 0.2,
          weight: options.weight || 2,
          opacity: options.opacity || 0.8
        });
      }
      
      // Add popup if provided
      if (options.popup) {
        overlay.bindPopup(options.popup);
      }
      
      // Add to map
      overlay.addTo(map);
      layers.drawing.addLayer(overlay);
      overlays.set(zoneId, overlay);
      
      console.log(`🔴 Zone overlay added: ${zoneId}`);
      return overlay;
      
    } catch (error) {
      console.error('❌ Failed to add zone overlay:', error);
      return null;
    }
  }
  
  /**
   * Remove zone overlay from map
   */
  export function removeZoneOverlay(mapOrContainerId, zoneId) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return false;
      
      const { instance: map, overlays, layers } = mapData;
      const overlay = overlays.get(zoneId);
      
      if (overlay) {
        map.removeLayer(overlay);
        layers.drawing.removeLayer(overlay);
        overlays.delete(zoneId);
        console.log(`🗑️ Zone overlay removed: ${zoneId}`);
        return true;
      }
      
      return false;
      
    } catch (error) {
      console.error('❌ Failed to remove zone overlay:', error);
      return false;
    }
  }
  
  /**
   * Export map as image
   */
  export function exportMapAsImage(mapOrContainerId, options = {}) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return null;
      
      const { instance: map } = mapData;
      
      // Use leaflet-image plugin if available
      if (typeof leafletImage !== 'undefined') {
        leafletImage(map, function(err, canvas) {
          if (err) {
            console.error('❌ Failed to export map image:', err);
            return;
          }
          
          // Convert to data URL
          const dataURL = canvas.toDataURL(options.format || 'image/png');
          
          // Download or callback
          if (options.download) {
            const link = document.createElement('a');
            link.download = options.filename || 'map-export.png';
            link.href = dataURL;
            link.click();
          }
          
          if (options.callback) {
            options.callback(dataURL);
          }
        });
      } else {
        console.warn('⚠️ leaflet-image plugin not available for map export');
      }
      
    } catch (error) {
      console.error('❌ Failed to export map:', error);
      return null;
    }
  }
  
  /**
   * Get map statistics
   */
  export function getMapStatistics(mapOrContainerId) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return null;
      
      const { markers, routes, overlays, instance: map } = mapData;
      
      // Count markers by type
      const markersByType = {};
      Object.keys(MAP_CONFIG.EMERGENCY_MARKERS).forEach(type => {
        markersByType[type] = 0;
      });
      
      // This would require storing incident data with markers
      // For now, just return total counts
      
      const stats = {
        totalMarkers: markers.size,
        totalRoutes: routes.size,
        totalOverlays: overlays.size,
        mapCenter: map.getCenter(),
        mapZoom: map.getZoom(),
        mapBounds: map.getBounds(),
        markersByType,
        lastUpdated: new Date().toISOString()
      };
      
      return stats;
      
    } catch (error) {
      console.error('❌ Failed to get map statistics:', error);
      return null;
    }
  }
  
  /**
   * Clear all map data
   */
  export function clearAllMapData(mapOrContainerId) {
    try {
      const result = {
        markers: clearAllEmergencyMarkers(mapOrContainerId),
        routes: clearAllRoutes(mapOrContainerId),
        overlays: clearAllOverlays(mapOrContainerId)
      };
      
      console.log('🧹 All map data cleared');
      return result;
      
    } catch (error) {
      console.error('❌ Failed to clear map data:', error);
      return false;
    }
  }
  
  /**
   * Clear all routes
   */
  function clearAllRoutes(mapOrContainerId) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return false;
      
      const { instance: map, routes, layers } = mapData;
      
      for (const route of routes.values()) {
        map.removeLayer(route.polyline);
        layers.drawing.removeLayer(route.polyline);
      }
      
      routes.clear();
      return true;
      
    } catch (error) {
      console.error('❌ Failed to clear routes:', error);
      return false;
    }
  }
  
  /**
   * Clear all overlays
   */
  function clearAllOverlays(mapOrContainerId) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return false;
      
      const { instance: map, overlays, layers } = mapData;
      
      for (const overlay of overlays.values()) {
        map.removeLayer(overlay);
        layers.drawing.removeLayer(overlay);
      }
      
      overlays.clear();
      return true;
      
    } catch (error) {
      console.error('❌ Failed to clear overlays:', error);
      return false;
    }
  }
  
  /**
   * Batch operations for performance
   */
  export function batchAddMarkers(mapOrContainerId, incidents) {
    try {
      const mapData = getMapData(mapOrContainerId);
      if (!mapData) return [];
      
      const { instance: map } = mapData;
      const addedMarkers = [];
      
      // Disable map updates during batch operation
      map._suspendEvents = true;
      
      for (const incident of incidents) {
        const marker = addEmergencyMarker(mapOrContainerId, incident);
        if (marker) {
          addedMarkers.push(marker);
        }
      }
      
      // Re-enable map updates
      map._suspendEvents = false;
      map.invalidateSize();
      
      console.log(`📍 Batch added ${addedMarkers.length} markers`);
      return addedMarkers;
      
    } catch (error) {
      console.error('❌ Failed to batch add markers:', error);
      return [];
    }
  }
  
  /**
   * Global utility functions for browser context
   */
  if (typeof window !== 'undefined') {
    // Make utility functions available globally for popup actions
    window.getDirections = function(coordinates) {
      if (userLocation && coordinates) {
        console.log(`🧭 Getting directions from ${userLocation} to ${coordinates}`);
        // This would integrate with a navigation app or show route on map
        
        // Example: open in Google Maps
        const url = `https://www.google.com/maps/dir/${userLocation[0]},${userLocation[1]}/${coordinates[0]},${coordinates[1]}`;
        window.open(url, '_blank');
      } else {
        console.warn('⚠️ User location or destination not available');
      }
    };
    
    window.updateIncident = function(incidentId) {
      console.log(`✏️ Update incident: ${incidentId}`);
      // This would open an update form or modal
      // Implementation depends on the application framework
    };
    
    window.shareIncident = function(incidentId) {
      console.log(`📤 Share incident: ${incidentId}`);
      // This would open sharing options
      if (navigator.share) {
        navigator.share({
          title: 'Emergency Incident',
          text: `Emergency incident ${incidentId}`,
          url: window.location.href
        });
      } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(`Emergency incident ${incidentId}: ${window.location.href}`);
        console.log('📋 Incident link copied to clipboard');
      }
    };
  }
  
  /**
   * CSS styles for map components (inject into document head)
   */
  const MAP_STYLES = `
  <style>
    .emergency-marker {
      background: none !important;
      border: none !important;
    }
    
    .emergency-popup .leaflet-popup-content {
      margin: 8px 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .emergency-popup-content {
      min-width: 250px;
    }
    
    .emergency-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #e5e7eb;
    }
    
    .emergency-header h3 {
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      flex: 1;
    }
    
    .emergency-icon {
      font-size: 20px;
    }
    
    .emergency-status {
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 500;
      text-transform: uppercase;
    }
    
    .status-active {
      background-color: #fee2e2;
      color: #dc2626;
    }
    
    .status-resolved {
      background-color: #d1fae5;
      color: #059669;
    }
    
    .status-pending {
      background-color: #fef3c7;
      color: #d97706;
    }
    
    .emergency-details p {
      margin: 4px 0;
      font-size: 14px;
    }
    
    .emergency-actions {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      padding-top: 8px;
      border-top: 1px solid #e5e7eb;
    }
    
    .emergency-actions button {
      padding: 6px 12px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      background: white;
      cursor: pointer;
      font-size: 12px;
      transition: all 0.2s;
    }
    
    .emergency-actions button:hover {
      background-color: #f3f4f6;
      border-color: #9ca3af;
    }
    
    .theme-selector {
      padding: 4px 8px;
      border: 1px solid #ccc;
      border-radius: 4px;
      background: white;
      font-size: 12px;
    }
    
    .location-btn {
      width: 30px;
      height: 30px;
      border: 2px solid rgba(0,0,0,0.2);
      background: white;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .location-btn:hover {
      background-color: #f9f9f9;
    }
    
    .emergency-layers-panel {
      background: white;
      padding: 12px;
      border-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border: 2px solid rgba(0,0,0,0.2);
      min-width: 160px;
    }
    
    .emergency-layers-panel h4 {
      margin: 0 0 8px 0;
      font-size: 14px;
      font-weight: 600;
    }
    
    .emergency-layers-panel label {
      display: block;
      margin: 4px 0;
      font-size: 12px;
      cursor: pointer;
    }
    
    .emergency-layers-panel input {
      margin-right: 6px;
    }
    
    .user-location-popup .leaflet-popup-content {
      margin: 8px 12px;
    }
    
    .route-popup .leaflet-popup-content {
      margin: 8px 12px;
    }
    
    @keyframes emergency-pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.1); }
      100% { transform: scale(1); }
    }
    
    /* Dark theme overrides */
    .leaflet-container.dark-theme {
      background: #1f2937;
    }
    
    .leaflet-container.dark-theme .leaflet-control-container {
      filter: invert(1) hue-rotate(180deg);
    }
  </style>
  `;
  
  /**
   * Initialize map styles
   */
  function initializeMapStyles() {
    if (typeof document !== 'undefined' && !document.getElementById('map-utils-styles')) {
      const styleElement = document.createElement('div');
      styleElement.id = 'map-utils-styles';
      styleElement.innerHTML = MAP_STYLES;
      document.head.appendChild(styleElement);
    }
  }
  
  // Initialize styles when module loads
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initializeMapStyles);
    } else {
      initializeMapStyles();
    }
  }
  
  // Export configuration for external access
  export { MAP_CONFIG };