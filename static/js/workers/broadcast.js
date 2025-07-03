// Enhanced broadcast.js - Advanced Emergency Broadcasting System
import {
  initP2PFallback,
  sendP2PBroadcast,
  isP2PActive,
  getConnectionStats,
  getConnectedPeers,
  onPeerConnected,
  onStatusUpdate
} from './fallback-webrtc.js';
import { updateBroadcastMap } from './broadcast-map.js';

// Configuration
const CONFIG = {
  POLL_INTERVAL: 15000,
  MAX_BROADCASTS: 100,
  MAX_CACHE_AGE: 3600000, // 1 hour
  PRIORITY_REFRESH_INTERVAL: 5000,
  EMERGENCY_POLL_INTERVAL: 3000,
  AUTO_CLEANUP_INTERVAL: 60000,
  BROADCAST_RETENTION: 24 * 60 * 60 * 1000, // 24 hours
  OFFLINE_STORAGE_KEY: 'emergency_broadcasts_cache',
  STATS_UPDATE_INTERVAL: 5000,
  MAX_MESSAGE_LENGTH: 500,
  PRIORITY_TYPES: ['critical', 'urgent', 'high', 'medium', 'low']
};

// Global state
let pollInterval = null;
let priorityPollInterval = null;
let statsInterval = null;
let cleanupInterval = null;
let broadcastCache = [];
let peerBroadcasts = [];
let isEmergencyMode = false;
let lastFetchTime = null;
let totalBroadcasts = 0;
let isInitialized = false;
let broadcastStats = {
  totalReceived: 0,
  totalSent: 0,
  peerMessages: 0,
  serverMessages: 0,
  lastUpdate: null
};

// Event emitter for broadcast events
class BroadcastEventEmitter {
  constructor() {
    this.events = {};
  }
  
  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }
  
  emit(event, data) {
    if (this.events[event]) {
      this.events[event].forEach(callback => callback(data));
    }
  }
}

const eventEmitter = new BroadcastEventEmitter();

/**
 * Enhanced broadcast listener initialization
 */
export function startBroadcastListener(options = {}) {
  if (isInitialized) {
    console.warn('⚠️ Broadcast listener already initialized');
    return;
  }
  
  console.log('🚀 Initializing enhanced emergency broadcast system');
  
  // Apply configuration overrides
  const config = { ...CONFIG, ...options };
  
  // Load cached broadcasts
  loadCachedBroadcasts();
  
  // Initialize P2P system
  initP2PFallback(handlePeerBroadcast, {
    onStatusChange: updateConnectionStatus
  });
  
  // Set up P2P event listeners
  onPeerConnected(handlePeerConnection);
  onStatusUpdate(handleP2PStatusUpdate);
  
  // Initial fetch and render
  fetchAndRenderBroadcasts();
  
  // Setup polling intervals
  setupPolling(config);
  
  // Setup diagnostics and monitoring
  setupDiagnostics();
  setupStatsMonitoring();
  setupAutoCleanup();
  
  // Setup emergency mode detection
  setupEmergencyModeDetection();
  
  // Setup offline support
  setupOfflineSupport();
  
  isInitialized = true;
  console.log('✅ Emergency broadcast system initialized');
}

/**
 * Setup polling intervals with dynamic adjustment
 */
function setupPolling(config) {
  // Standard polling
  pollInterval = setInterval(() => {
    if (!isP2PActive() || isEmergencyMode) {
      fetchAndRenderBroadcasts();
    }
  }, isEmergencyMode ? CONFIG.EMERGENCY_POLL_INTERVAL : config.POLL_INTERVAL);
  
  // Priority message polling (always active)
  priorityPollInterval = setInterval(() => {
    fetchPriorityBroadcasts();
  }, CONFIG.PRIORITY_REFRESH_INTERVAL);
  
  console.log(`⏰ Polling setup: Standard ${config.POLL_INTERVAL}ms, Priority ${CONFIG.PRIORITY_REFRESH_INTERVAL}ms`);
}

/**
 * Enhanced broadcast fetching with error handling and caching
 */
async function fetchAndRenderBroadcasts() {
  try {
    console.log('📡 Fetching broadcasts from server');
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);
    
    const response = await fetch('/api/broadcasts', {
      signal: controller.signal,
      headers: {
        'Cache-Control': 'no-cache',
        'Last-Modified-Since': lastFetchTime ? new Date(lastFetchTime).toUTCString() : ''
      }
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const broadcasts = await response.json();
    
    // Process and cache broadcasts
    processServerBroadcasts(broadcasts);
    lastFetchTime = Date.now();
    
    // Update stats
    broadcastStats.serverMessages = broadcasts.length;
    broadcastStats.lastUpdate = new Date().toLocaleTimeString();
    
    renderAllBroadcasts();
    
    console.log(`📥 Fetched ${broadcasts.length} broadcasts from server`);
    
  } catch (error) {
    console.error('❌ Failed to fetch broadcasts:', error);
    
    // Use cached data if available
    if (broadcastCache.length > 0) {
      console.log('📦 Using cached broadcasts');
      renderAllBroadcasts();
    } else {
      showBroadcastError('Unable to load emergency broadcasts. Check your connection.');
    }
  }
}

/**
 * Fetch high-priority broadcasts more frequently
 */
async function fetchPriorityBroadcasts() {
  try {
    const response = await fetch('/api/broadcasts/priority', {
      method: 'GET',
      headers: { 'Cache-Control': 'no-cache' }
    });
    
    if (response.ok) {
      const priorityBroadcasts = await response.json();
      
      if (priorityBroadcasts.length > 0) {
        console.log(`🚨 Received ${priorityBroadcasts.length} priority broadcasts`);
        
        priorityBroadcasts.forEach(broadcast => {
          broadcast.source = 'server_priority';
          broadcast.timestamp = new Date().toISOString();
        });
        
        // Add to cache and trigger emergency mode
        processServerBroadcasts(priorityBroadcasts);
        
        if (!isEmergencyMode) {
          activateEmergencyMode();
        }
        
        renderAllBroadcasts();
      }
    }
  } catch (error) {
    console.warn('⚠️ Priority broadcast fetch failed:', error);
  }
}

/**
 * Process broadcasts from server
 */
function processServerBroadcasts(broadcasts) {
  const now = Date.now();
  
  // Add metadata and validate
  const processedBroadcasts = broadcasts
    .filter(broadcast => broadcast && broadcast.message)
    .map(broadcast => ({
      id: broadcast.id || generateBroadcastId(),
      message: broadcast.message,
      severity: broadcast.severity || 'medium',
      tone: broadcast.tone || 'descriptive',
      escalation: broadcast.escalation || 'medium',
      sentiment: broadcast.sentiment || 'neutral',
      location: broadcast.location,
      timestamp: broadcast.timestamp || new Date().toISOString(),
      source: 'server',
      priority: broadcast.priority || determinePriority(broadcast),
      receivedAt: now,
      ...broadcast
    }))
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  
  // Merge with existing cache, avoiding duplicates
  const existingIds = new Set(broadcastCache.map(b => b.id));
  const newBroadcasts = processedBroadcasts.filter(b => !existingIds.has(b.id));
  
  if (newBroadcasts.length > 0) {
    broadcastCache = [...newBroadcasts, ...broadcastCache]
      .slice(0, CONFIG.MAX_BROADCASTS);
    
    // Cache for offline use
    cacheBroadcasts();
    
    // Check for emergency conditions
    checkEmergencyConditions(newBroadcasts);
  }
  
  broadcastStats.totalReceived += newBroadcasts.length;
}

/**
 * Enhanced peer broadcast handler
 */
function handlePeerBroadcast(data) {
  if (!data || !data.message) {
    console.warn('⚠️ Invalid peer broadcast data received');
    return;
  }
  
  console.log('📡 Peer broadcast received:', data.type || 'general');
  
  const peerBroadcast = {
    id: data.id || generateBroadcastId(),
    message: data.message,
    severity: data.severity || 'medium',
    tone: data.tone || 'urgent',
    escalation: data.escalation || 'medium', 
    sentiment: data.sentiment || 'concerned',
    location: data.location,
    timestamp: data.timestamp || new Date().toISOString(),
    source: 'peer',
    fromPeer: data.fromPeer || 'unknown',
    relayCount: data.relayCount || 0,
    priority: data.priority || determinePriority(data),
    receivedAt: Date.now(),
    ...data
  };
  
  // Add to peer broadcasts cache
  peerBroadcasts.unshift(peerBroadcast);
  peerBroadcasts = peerBroadcasts.slice(0, 50); // Keep last 50 peer messages
  
  // Update stats
  broadcastStats.peerMessages++;
  broadcastStats.totalReceived++;
  
  // Check for emergency conditions
  if (isEmergencyPriority(peerBroadcast.priority)) {
    activateEmergencyMode();
  }
  
  // Render updated broadcasts
  renderAllBroadcasts();
  
  // Show notification for high-priority peer messages
  if (peerBroadcast.priority === 'critical' || peerBroadcast.priority === 'urgent') {
    showBroadcastNotification(peerBroadcast);
  }
  
  // Emit event for external handlers
  eventEmitter.emit('peer_broadcast', peerBroadcast);
}

/**
 * Enhanced broadcast rendering with better UX
 */
function renderAllBroadcasts() {
  const feed = document.getElementById('broadcast-feed');
  const alert = document.getElementById('broadcast-alert');
  
  if (!feed) {
    console.warn('⚠️ Broadcast feed element not found');
    return;
  }
  
  // Combine and sort all broadcasts
  const allBroadcasts = [...peerBroadcasts, ...broadcastCache]
    .sort((a, b) => {
      // Sort by priority first, then by timestamp
      const priorityOrder = { critical: 0, urgent: 1, high: 2, medium: 3, low: 4 };
      const aPriority = priorityOrder[a.priority] || 3;
      const bPriority = priorityOrder[b.priority] || 3;
      
      if (aPriority !== bPriority) {
        return aPriority - bPriority;
      }
      
      return new Date(b.timestamp) - new Date(a.timestamp);
    })
    .slice(0, CONFIG.MAX_BROADCASTS);
  
  totalBroadcasts = allBroadcasts.length;
  
  if (allBroadcasts.length === 0) {
    feed.innerHTML = `
      <div class="empty-broadcasts">
        <div class="empty-icon">📡</div>
        <h3>No Active Broadcasts</h3>
        <p>Emergency broadcasts will appear here when available.</p>
      </div>
    `;
    
    if (alert) alert.style.display = 'none';
    return;
  }
  
  // Render broadcast cards
  const cardsHTML = allBroadcasts.map(renderBroadcastCard).join('');
  feed.innerHTML = cardsHTML;
  
  // Show/hide alert banner
  const hasHighPriority = allBroadcasts.some(b => 
    b.priority === 'critical' || b.priority === 'urgent'
  );
  
  if (alert) {
    alert.style.display = hasHighPriority ? 'block' : 'none';
    
    if (hasHighPriority) {
      const criticalCount = allBroadcasts.filter(b => b.priority === 'critical').length;
      const urgentCount = allBroadcasts.filter(b => b.priority === 'urgent').length;
      
      alert.innerHTML = `
        🚨 <strong>EMERGENCY ALERTS ACTIVE</strong> - 
        ${criticalCount} Critical, ${urgentCount} Urgent broadcasts
      `;
    }
  }
  
  // Update map if available
  try {
    updateBroadcastMap(allBroadcasts);
  } catch (error) {
    console.warn('⚠️ Map update failed:', error);
  }
  
  // Update UI counters
  updateBroadcastCounters(allBroadcasts);
  
  console.log(`📺 Rendered ${allBroadcasts.length} broadcasts`);
}

/**
 * Enhanced broadcast card rendering
 */
function renderBroadcastCard(broadcast) {
  const priorityBadge = getPriorityBadge(broadcast.priority);
  const toneBadge = getToneBadge(broadcast.tone);
  const escalationBadge = getEscalationBadge(broadcast.escalation);
  const sentimentIcon = getSentimentIcon(broadcast.sentiment);
  const sourceBadge = getSourceBadge(broadcast.source);
  const timeAgo = getTimeAgo(broadcast.timestamp);
  
  const cardClass = `broadcast-card priority-${broadcast.priority} source-${broadcast.source}`;
  
  return `
    <div class="${cardClass}" data-broadcast-id="${broadcast.id}" data-priority="${broadcast.priority}">
      <div class="broadcast-header">
        <div class="broadcast-meta">
          ${sentimentIcon} ${priorityBadge} ${sourceBadge}
          ${broadcast.location ? `<span class="location-badge">📍 ${broadcast.location}</span>` : ''}
        </div>
        <div class="broadcast-time" title="${broadcast.timestamp}">
          ${timeAgo}
        </div>
      </div>
      
      <div class="broadcast-content">
        <div class="broadcast-severity">
          <strong>${broadcast.severity.toUpperCase()}</strong>
        </div>
        <div class="broadcast-message">
          ${escapeHtml(broadcast.message)}
        </div>
      </div>
      
      <div class="broadcast-footer">
        <div class="broadcast-badges">
          ${toneBadge} ${escalationBadge}
        </div>
        <div class="broadcast-actions">
          ${broadcast.source === 'peer' ? `<span class="relay-info">Relayed ${broadcast.relayCount || 0}x</span>` : ''}
          <button class="share-btn" onclick="shareBroadcast('${broadcast.id}')" title="Share this broadcast">📤</button>
        </div>
      </div>
    </div>
  `;
}

/**
 * Enhanced badge generation functions
 */
function getPriorityBadge(priority) {
  const priorityConfig = {
    critical: { color: '#dc2626', icon: '🔴', label: 'CRITICAL' },
    urgent: { color: '#f59e0b', icon: '🟡', label: 'URGENT' },
    high: { color: '#3b82f6', icon: '🔵', label: 'HIGH' },
    medium: { color: '#10b981', icon: '🟢', label: 'MEDIUM' },
    low: { color: '#6b7280', icon: '⚪', label: 'LOW' }
  };
  
  const config = priorityConfig[priority] || priorityConfig.medium;
  
  return `<span class="priority-badge priority-${priority}" style="background-color: ${config.color};">
    ${config.icon} ${config.label}
  </span>`;
}

function getToneBadge(tone) {
  const toneConfig = {
    descriptive: { color: '#6b7280', label: 'Descriptive' },
    urgent: { color: '#f59e0b', label: 'Urgent' },
    frantic: { color: '#dc2626', label: 'Frantic' },
    helpless: { color: '#7c3aed', label: 'Helpless' },
    calm: { color: '#10b981', label: 'Calm' }
  };
  
  if (!tone) return '';
  
  const config = toneConfig[tone.toLowerCase()] || { color: '#6b7280', label: tone };
  
  return `<span class="tone-badge" style="background-color: ${config.color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">
    ${config.label}
  </span>`;
}

function getEscalationBadge(level) {
  const escalationMap = {
    low: '🟢 Low',
    moderate: '🟡 Moderate', 
    high: '🟠 High',
    critical: '🔴 Critical'
  };
  
  if (!level) return '';
  
  const badge = escalationMap[level.toLowerCase()] || level;
  
  return `<span class="escalation-badge" style="margin-left: 8px; font-weight: bold; color: #374151;">
    ${badge}
  </span>`;
}

function getSentimentIcon(sentiment) {
  const sentimentMap = {
    calm: '🧘',
    neutral: '😐',
    concerned: '🤔',
    worried: '😟',
    anxious: '😰',
    panic: '🚨',
    distressed: '😢'
  };
  
  return sentimentMap[sentiment?.toLowerCase()] || '📢';
}

function getSourceBadge(source) {
  const sourceConfig = {
    server: { color: '#3b82f6', icon: '🌐', label: 'Server' },
    peer: { color: '#10b981', icon: '📡', label: 'P2P' },
    server_priority: { color: '#dc2626', icon: '🚨', label: 'Emergency' },
    manual: { color: '#7c3aed', icon: '👤', label: 'Manual' }
  };
  
  const config = sourceConfig[source] || { color: '#6b7280', icon: '❓', label: 'Unknown' };
  
  return `<span class="source-badge" style="background-color: ${config.color}; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem; font-weight: bold;">
    ${config.icon} ${config.label}
  </span>`;
}

/**
 * Enhanced broadcast submission with validation
 */
export function submitBroadcast(payload, options = {}) {
  if (!payload || !payload.message) {
    console.error('❌ Invalid broadcast payload');
    return false;
  }
  
  // Validate message length
  if (payload.message.length > CONFIG.MAX_MESSAGE_LENGTH) {
    console.warn(`⚠️ Message too long: ${payload.message.length} characters`);
    payload.message = payload.message.substring(0, CONFIG.MAX_MESSAGE_LENGTH) + '...';
  }
  
  // Add metadata
  const broadcastData = {
    id: generateBroadcastId(),
    timestamp: new Date().toISOString(),
    priority: payload.priority || 'medium',
    source: 'local',
    ...payload
  };
  
  console.log(`📤 Submitting broadcast: ${broadcastData.priority} priority`);
  
  let submitted = false;
  
  // Try P2P first for urgent messages
  if (isP2PActive() && (options.preferP2P || isEmergencyPriority(broadcastData.priority))) {
    try {
      sendP2PBroadcast(broadcastData, {
        type: broadcastData.priority === 'critical' ? 'emergency_alert' : 'broadcast',
        priority: broadcastData.priority,
        reliability: options.reliability || 'best-effort'
      });
      
      submitted = true;
      broadcastStats.totalSent++;
      
      console.log('📡 Broadcast sent via P2P');
    } catch (error) {
      console.error('❌ P2P broadcast failed:', error);
    }
  }
  
  // Also send to server (redundancy for critical messages)
  if (!isP2PActive() || broadcastData.priority === 'critical' || options.sendToServer !== false) {
    fetch('/api/broadcast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(broadcastData)
    })
    .then(response => {
      if (response.ok) {
        console.log('🌐 Broadcast sent to server');
        broadcastStats.totalSent++;
        submitted = true;
      } else {
        throw new Error(`Server rejected broadcast: ${response.status}`);
      }
    })
    .catch(error => {
      console.error('❌ Server broadcast failed:', error);
      
      // Queue for offline sync if available
      if (window.syncQueue) {
        window.syncQueue.queueReport(broadcastData, {
          type: 'emergency_broadcast',
          priority: broadcastData.priority
        });
      }
    });
  }
  
  return submitted;
}

/**
 * Setup enhanced diagnostics with real-time updates
 */
function setupDiagnostics() {
  const diagnosticElements = [
    'mesh-mode',
    'peer-count', 
    'broadcast-count',
    'connection-status',
    'last-update'
  ];
  
  setInterval(() => {
    const connectedPeers = getConnectedPeers();
    const connectionStats = getConnectionStats();
    
    // Update mesh status
    const meshStatus = document.getElementById('mesh-mode');
    if (meshStatus) {
      meshStatus.textContent = isP2PActive() 
        ? `Connected (${connectedPeers.length} peers)`
        : 'Disconnected';
      meshStatus.className = isP2PActive() ? 'status-connected' : 'status-disconnected';
    }
    
    // Update peer count
    const peerCount = document.getElementById('peer-count');
    if (peerCount) {
      peerCount.textContent = connectedPeers.length;
    }
    
    // Update broadcast count
    const broadcastCount = document.getElementById('broadcast-count');
    if (broadcastCount) {
      broadcastCount.textContent = totalBroadcasts;
    }
    
    // Update connection status
    const connectionStatus = document.getElementById('connection-status');
    if (connectionStatus) {
      const status = navigator.onLine ? 'Online' : 'Offline';
      connectionStatus.textContent = status;
      connectionStatus.className = navigator.onLine ? 'status-online' : 'status-offline';
    }
    
    // Update last update time
    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate) {
      lastUpdate.textContent = broadcastStats.lastUpdate || 'Never';
    }
    
  }, 3000);
}

/**
 * Setup statistics monitoring
 */
function setupStatsMonitoring() {
  statsInterval = setInterval(() => {
    const stats = {
      ...broadcastStats,
      connectedPeers: getConnectedPeers().length,
      isP2PActive: isP2PActive(),
      isEmergencyMode: isEmergencyMode,
      totalBroadcasts: totalBroadcasts,
      cacheSize: broadcastCache.length,
      peerCacheSize: peerBroadcasts.length
    };
    
    // Emit stats for external monitoring
    eventEmitter.emit('stats_update', stats);
    
    // Update stats display if element exists
    const statsDisplay = document.getElementById('broadcast-stats');
    if (statsDisplay) {
      statsDisplay.innerHTML = `
        <div class="stat-item">📥 Received: ${stats.totalReceived}</div>
        <div class="stat-item">📤 Sent: ${stats.totalSent}</div>
        <div class="stat-item">👥 Peers: ${stats.connectedPeers}</div>
        <div class="stat-item">📡 P2P: ${stats.isP2PActive ? 'Active' : 'Inactive'}</div>
      `;
    }
    
  }, CONFIG.STATS_UPDATE_INTERVAL);
}

/**
 * Setup automatic cleanup of old broadcasts
 */
function setupAutoCleanup() {
  cleanupInterval = setInterval(() => {
    const now = Date.now();
    const cutoff = now - CONFIG.BROADCAST_RETENTION;
    
    // Clean up server broadcasts
    const oldCacheSize = broadcastCache.length;
    broadcastCache = broadcastCache.filter(broadcast => {
      const broadcastTime = new Date(broadcast.timestamp).getTime();
      return broadcastTime > cutoff;
    });
    
    // Clean up peer broadcasts (shorter retention)
    const peerCutoff = now - (CONFIG.BROADCAST_RETENTION / 2);
    const oldPeerSize = peerBroadcasts.length;
    peerBroadcasts = peerBroadcasts.filter(broadcast => {
      const broadcastTime = new Date(broadcast.timestamp).getTime();
      return broadcastTime > peerCutoff;
    });
    
    const cleanedItems = (oldCacheSize - broadcastCache.length) + (oldPeerSize - peerBroadcasts.length);
    
    if (cleanedItems > 0) {
      console.log(`🧹 Cleaned up ${cleanedItems} old broadcasts`);
      cacheBroadcasts(); // Update cache
    }
    
  }, CONFIG.AUTO_CLEANUP_INTERVAL);
}

/**
 * Emergency mode detection and handling
 */
function setupEmergencyModeDetection() {
  // Check for emergency keywords in broadcasts
  const emergencyKeywords = [
    'fire', 'explosion', 'medical emergency', 'evacuation', 
    'tsunami', 'earthquake', 'tornado', 'flood', 'chemical spill',
    'active shooter', 'bomb', 'terrorist', 'critical', 'urgent'
  ];
  
  eventEmitter.on('peer_broadcast', (broadcast) => {
    const message = broadcast.message.toLowerCase();
    const hasEmergencyKeyword = emergencyKeywords.some(keyword => 
      message.includes(keyword)
    );
    
    if (hasEmergencyKeyword || isEmergencyPriority(broadcast.priority)) {
      activateEmergencyMode();
    }
  });
}

function activateEmergencyMode() {
  if (isEmergencyMode) return;
  
  console.log('🚨 EMERGENCY MODE ACTIVATED');
  isEmergencyMode = true;
  
  // Increase polling frequency
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = setInterval(() => {
      if (!isP2PActive() || isEmergencyMode) {
        fetchAndRenderBroadcasts();
      }
    }, CONFIG.EMERGENCY_POLL_INTERVAL);
  }
  
  // Update UI to show emergency mode
  const emergencyIndicators = document.querySelectorAll('.emergency-mode-indicator');
  emergencyIndicators.forEach(indicator => {
    indicator.style.display = 'block';
    indicator.textContent = '🚨 EMERGENCY MODE ACTIVE';
  });
  
  // Auto-deactivate after 30 minutes
  setTimeout(() => {
    deactivateEmergencyMode();
  }, 30 * 60 * 1000);
  
  eventEmitter.emit('emergency_mode_activated');
}

function deactivateEmergencyMode() {
  console.log('✅ Emergency mode deactivated');
  isEmergencyMode = false;
  
  // Reset polling frequency
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = setInterval(() => {
      if (!isP2PActive() || isEmergencyMode) {
        fetchAndRenderBroadcasts();
      }
    }, CONFIG.POLL_INTERVAL);
  }
  
  // Update UI
  const emergencyIndicators = document.querySelectorAll('.emergency-mode-indicator');
  emergencyIndicators.forEach(indicator => {
    indicator.style.display = 'none';
  });
  
  eventEmitter.emit('emergency_mode_deactivated');
}

/**
 * Offline support and caching
 */
function setupOfflineSupport() {
  // Handle online/offline events
  window.addEventListener('online', () => {
    console.log('🌐 Connection restored - refreshing broadcasts');
    fetchAndRenderBroadcasts();
  });
  
  window.addEventListener('offline', () => {
    console.log('📴 Connection lost - using cached broadcasts');
    renderAllBroadcasts();
  });
}

function loadCachedBroadcasts() {
  try {
    const cached = localStorage.getItem(CONFIG.OFFLINE_STORAGE_KEY);
    if (cached) {
      const data = JSON.parse(cached);
      broadcastCache = data.broadcasts || [];
      broadcastStats = { ...broadcastStats, ...data.stats };
      
      console.log(`📦 Loaded ${broadcastCache.length} cached broadcasts`);
    }
  } catch (error) {
    console.warn('⚠️ Failed to load cached broadcasts:', error);
  }
}

function cacheBroadcasts() {
  try {
    const cacheData = {
      broadcasts: broadcastCache.slice(0, 50), // Limit cache size
      stats: broadcastStats,
      timestamp: Date.now()
    };
    
    localStorage.setItem(CONFIG.OFFLINE_STORAGE_KEY, JSON.stringify(cacheData));
  } catch (error) {
    console.warn('⚠️ Failed to cache broadcasts:', error);
  }
}

/**
 * Utility functions
 */
function determinePriority(broadcast) {
  if (!broadcast) return 'medium';
  
  // Check explicit priority first
  if (broadcast.priority && CONFIG.PRIORITY_TYPES.includes(broadcast.priority)) {
    return broadcast.priority;
  }
  
  // Determine priority based on content
  const message = (broadcast.message || '').toLowerCase();
  const severity = (broadcast.severity || '').toLowerCase();
  const escalation = (broadcast.escalation || '').toLowerCase();
  
  // Critical indicators
  const criticalKeywords = ['fire', 'explosion', 'medical emergency', 'evacuation', 'critical', 'life threatening'];
  const urgentKeywords = ['urgent', 'immediate', 'help', 'emergency', 'danger'];
  
  if (escalation === 'critical' || severity === 'critical' || 
      criticalKeywords.some(keyword => message.includes(keyword))) {
    return 'critical';
  }
  
  if (escalation === 'high' || severity === 'urgent' ||
      urgentKeywords.some(keyword => message.includes(keyword))) {
    return 'urgent';
  }
  
  if (escalation === 'moderate' || severity === 'high') {
    return 'high';
  }
  
  return 'medium';
}

function isEmergencyPriority(priority) {
  return priority === 'critical' || priority === 'urgent';
}

function checkEmergencyConditions(broadcasts) {
  const emergencyBroadcasts = broadcasts.filter(b => isEmergencyPriority(b.priority));
  
  if (emergencyBroadcasts.length > 0) {
    activateEmergencyMode();
  }
}

function generateBroadcastId() {
  return `broadcast_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function getTimeAgo(timestamp) {
  const now = Date.now();
  const time = new Date(timestamp).getTime();
  const diff = now - time;
  
  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return `${Math.floor(diff / 86400000)}d ago`;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function updateBroadcastCounters(broadcasts) {
  // Update various counter elements
  const counterElements = [
    { id: 'total-broadcasts', value: broadcasts.length },
    { id: 'critical-broadcasts', value: broadcasts.filter(b => b.priority === 'critical').length },
    { id: 'urgent-broadcasts', value: broadcasts.filter(b => b.priority === 'urgent').length },
    { id: 'peer-broadcasts', value: broadcasts.filter(b => b.source === 'peer').length },
    { id: 'server-broadcasts', value: broadcasts.filter(b => b.source === 'server').length }
  ];
  
  counterElements.forEach(({ id, value }) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  });
}

function showBroadcastError(message) {
  const feed = document.getElementById('broadcast-feed');
  if (feed) {
    feed.innerHTML = `
      <div class="broadcast-error">
        <div class="error-icon">⚠️</div>
        <h3>Unable to Load Broadcasts</h3>
        <p>${message}</p>
        <button onclick="window.location.reload()" class="retry-button">
          🔄 Retry
        </button>
      </div>
    `;
  }
}

function showBroadcastNotification(broadcast) {
  // Show browser notification if permitted
  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification(`Emergency Alert: ${broadcast.severity}`, {
      body: broadcast.message.substring(0, 100) + (broadcast.message.length > 100 ? '...' : ''),
      icon: '/static/icons/emergency-icon.png',
      tag: broadcast.id,
      requireInteraction: true
    });
  }
  
  // Also show in-page notification
  const notification = document.createElement('div');
  notification.className = 'broadcast-notification priority-' + broadcast.priority;
  notification.innerHTML = `
    <div class="notification-header">
      <strong>🚨 ${broadcast.priority.toUpperCase()} ALERT</strong>
      <button onclick="this.parentElement.parentElement.remove()">×</button>
    </div>
    <div class="notification-body">${broadcast.message}</div>
  `;
  
  document.body.appendChild(notification);
  
  // Auto-remove after 10 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.remove();
    }
  }, 10000);
}

/**
 * Event handlers
 */
function handlePeerConnection(data) {
  console.log(`👥 New peer connected: ${data.peerId}`);
  
  // Send introduction broadcast
  setTimeout(() => {
    sendP2PBroadcast({
      type: 'peer_introduction',
      message: 'Emergency response system online and ready to relay broadcasts',
      severity: 'info',
      priority: 'low'
    });
  }, 2000);
}

function handleP2PStatusUpdate(data) {
  // Update connection status displays
  const statusElements = document.querySelectorAll('.p2p-status');
  statusElements.forEach(element => {
    element.textContent = data.status;
    element.className = `p2p-status ${data.type}`;
  });
}

function updateConnectionStatus(isConnected) {
  const statusElements = document.querySelectorAll('.connection-status');
  statusElements.forEach(element => {
    element.textContent = isConnected ? 'Connected' : 'Disconnected';
    element.className = `connection-status ${isConnected ? 'connected' : 'disconnected'}`;
  });
}

/**
 * Global functions for UI interaction
 */
window.shareBroadcast = function(broadcastId) {
  const broadcast = [...broadcastCache, ...peerBroadcasts].find(b => b.id === broadcastId);
  
  if (!broadcast) {
    console.warn('⚠️ Broadcast not found for sharing:', broadcastId);
    return;
  }
  
  // Re-broadcast via P2P if possible
  if (isP2PActive()) {
    sendP2PBroadcast({
      ...broadcast,
      relayCount: (broadcast.relayCount || 0) + 1,
      relayedBy: 'user',
      relayTimestamp: new Date().toISOString()
    });
    
    console.log('📡 Broadcast shared via P2P');
    
    // Show feedback
    const button = document.querySelector(`[onclick="shareBroadcast('${broadcastId}')"]`);
    if (button) {
      const originalText = button.innerHTML;
      button.innerHTML = '✅';
      button.disabled = true;
      
      setTimeout(() => {
        button.innerHTML = originalText;
        button.disabled = false;
      }, 2000);
    }
  } else {
    // Fallback to copying to clipboard
    const shareText = `EMERGENCY ALERT: ${broadcast.message} (${broadcast.timestamp})`;
    navigator.clipboard.writeText(shareText).then(() => {
      console.log('📋 Broadcast copied to clipboard');
    });
  }
};

window.filterBroadcasts = function(priority) {
  const cards = document.querySelectorAll('.broadcast-card');
  
  cards.forEach(card => {
    if (!priority || card.dataset.priority === priority) {
      card.style.display = 'block';
    } else {
      card.style.display = 'none';
    }
  });
};

window.clearBroadcasts = function() {
  if (confirm('Are you sure you want to clear all broadcasts? This action cannot be undone.')) {
    broadcastCache = [];
    peerBroadcasts = [];
    totalBroadcasts = 0;
    
    // Clear cache
    localStorage.removeItem(CONFIG.OFFLINE_STORAGE_KEY);
    
    renderAllBroadcasts();
    console.log('🧹 All broadcasts cleared');
  }
};

/**
 * Public API for external use
 */
export function getBroadcastStats() {
  return {
    ...broadcastStats,
    totalBroadcasts: totalBroadcasts,
    isEmergencyMode: isEmergencyMode,
    connectedPeers: getConnectedPeers().length
  };
}

export function getAllBroadcasts() {
  return [...peerBroadcasts, ...broadcastCache];
}

export function getBroadcastById(id) {
  return [...peerBroadcasts, ...broadcastCache].find(b => b.id === id);
}

export function addManualBroadcast(broadcastData) {
  const broadcast = {
    id: generateBroadcastId(),
    timestamp: new Date().toISOString(),
    source: 'manual',
    priority: determinePriority(broadcastData),
    ...broadcastData
  };
  
  peerBroadcasts.unshift(broadcast);
  renderAllBroadcasts();
  
  return broadcast.id;
}

export function subscribeToUpdates(callback) {
  eventEmitter.on('broadcast_update', callback);
}

export function subscribeToStats(callback) {
  eventEmitter.on('stats_update', callback);
}

export function forceEmergencyMode() {
  activateEmergencyMode();
}

export function getEmergencyMode() {
  return isEmergencyMode;
}

/**
 * Cleanup function
 */
export function stopBroadcastListener() {
  console.log('🛑 Stopping broadcast listener');
  
  // Clear intervals
  if (pollInterval) clearInterval(pollInterval);
  if (priorityPollInterval) clearInterval(priorityPollInterval);
  if (statsInterval) clearInterval(statsInterval);
  if (cleanupInterval) clearInterval(cleanupInterval);
  
  // Cache current state
  cacheBroadcasts();
  
  isInitialized = false;
  console.log('✅ Broadcast listener stopped');
}

// Initialize notification permissions
if ('Notification' in window && Notification.permission === 'default') {
  Notification.requestPermission().then(permission => {
    console.log(`📢 Notification permission: ${permission}`);
  });
}

// Auto-cleanup on page unload
window.addEventListener('beforeunload', stopBroadcastListener);

console.log('🚀 Enhanced emergency broadcast system loaded');