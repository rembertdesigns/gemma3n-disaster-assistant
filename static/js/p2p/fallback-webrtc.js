// Enhanced static/js/p2p/fallback-webrtc.js - Emergency P2P Communication System
import SimplePeer from 'simple-peer';
import io from 'socket.io-client';
import QRCode from 'qrcode';

// Configuration
const CONFIG = {
  SOCKET_URL: window.location.origin.replace(/^http/, 'ws') || 'ws://localhost:3001',
  SOCKET_RECONNECT_ATTEMPTS: 5,
  SOCKET_RECONNECT_DELAY: 2000,
  PEER_RECONNECT_ATTEMPTS: 3,
  PEER_RECONNECT_DELAY: 1000,
  MESSAGE_TIMEOUT: 30000,
  HEARTBEAT_INTERVAL: 15000,
  MAX_PEERS: 10,
  MAX_MESSAGE_SIZE: 1024 * 1024, // 1MB
  PRIORITY_TYPES: ['emergency_alert', 'medical_emergency', 'evacuation_order'],
  QR_CODE_SIZE: 256,
  AUTO_CLEANUP_INTERVAL: 60000, // 1 minute
  STATS_UPDATE_INTERVAL: 5000
};

// Global state
let peers = new Map();
let peerConnections = new Map();
let messageQueue = [];
let p2pActive = false;
let externalCallback = null;
let socket = null;
let manualPeer = null;
let qrRendered = false;
let isInitialized = false;
let localPeerId = null;
let connectionStats = {
  totalConnections: 0,
  activeConnections: 0,
  messagesSent: 0,
  messagesReceived: 0,
  bytesTransferred: 0,
  lastActivity: null,
  uptime: Date.now()
};

// Event emitter for internal communication
class P2PEventEmitter {
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
  
  off(event, callback) {
    if (this.events[event]) {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    }
  }
}

const eventEmitter = new P2PEventEmitter();

/**
 * Enhanced P2P initialization with robust error handling
 */
export function initP2PFallback(onMessage, options = {}) {
  if (isInitialized) {
    console.warn('⚠️ P2P system already initialized');
    return;
  }
  
  console.log('🚀 Initializing enhanced P2P emergency communication system');
  
  externalCallback = onMessage;
  localPeerId = generatePeerId();
  isInitialized = true;
  
  // Apply configuration overrides
  Object.assign(CONFIG, options);
  
  try {
    initializeSocket();
    setupHeartbeat();
    setupAutoCleanup();
    setupStatsUpdates();
    
    console.log(`✅ P2P system initialized with ID: ${localPeerId}`);
    
    // Auto-generate manual pairing on load
    setTimeout(manualSignalOffer, 1000);
    
  } catch (error) {
    console.error('❌ P2P initialization failed:', error);
    updateDiagnostics('Initialization Failed', 'error');
  }
}

/**
 * Enhanced socket connection with reconnection logic
 */
function initializeSocket() {
  let reconnectAttempts = 0;
  
  function connectSocket() {
    console.log(`🔌 Connecting to signaling server: ${CONFIG.SOCKET_URL}`);
    
    socket = io(CONFIG.SOCKET_URL, {
      transports: ['websocket', 'polling'],
      timeout: 10000,
      reconnection: true,
      reconnectionAttempts: CONFIG.SOCKET_RECONNECT_ATTEMPTS,
      reconnectionDelay: CONFIG.SOCKET_RECONNECT_DELAY
    });
    
    socket.on('connect', () => {
      console.log('📡 Connected to signaling server');
      reconnectAttempts = 0;
      updateDiagnostics('Signaling Connected', 'success');
      
      // Register with server
      socket.emit('register', {
        peerId: localPeerId,
        type: 'emergency_client',
        capabilities: ['mesh_relay', 'emergency_broadcast'],
        timestamp: Date.now()
      });
    });
    
    socket.on('disconnect', (reason) => {
      console.warn(`📡 Disconnected from signaling server: ${reason}`);
      updateDiagnostics('Signaling Disconnected', 'warning');
    });
    
    socket.on('connect_error', (error) => {
      reconnectAttempts++;
      console.error(`❌ Signaling connection error (attempt ${reconnectAttempts}):`, error);
      
      if (reconnectAttempts >= CONFIG.SOCKET_RECONNECT_ATTEMPTS) {
        console.log('🔄 Switching to manual-only mode');
        updateDiagnostics('Manual Mode Only', 'warning');
      }
    });
    
    socket.on('signal', handleIncomingSignal);
    socket.on('peer_list', handlePeerList);
    socket.on('peer_disconnect', handlePeerDisconnect);
    socket.on('emergency_broadcast', handleEmergencyBroadcast);
  }
  
  // Try to connect, fallback to manual mode if failed
  try {
    connectSocket();
  } catch (error) {
    console.warn('⚠️ Socket connection failed, continuing in manual-only mode');
    updateDiagnostics('Manual Mode Only', 'warning');
  }
}

/**
 * Handle incoming WebRTC signals
 */
function handleIncomingSignal({ from, data, type }) {
  try {
    console.log(`📥 Received signal from ${from}: ${type || 'offer/answer'}`);
    
    if (!peers.has(from)) {
      createPeerConnection(from, false);
    }
    
    const peer = peers.get(from);
    if (peer && !peer.destroyed) {
      peer.signal(data);
    }
    
  } catch (error) {
    console.error('❌ Error handling incoming signal:', error);
  }
}

/**
 * Handle peer list updates from server
 */
function handlePeerList(peerList) {
  console.log(`👥 Received peer list: ${peerList.length} peers available`);
  
  peerList.forEach(peerId => {
    if (peerId !== localPeerId && !peers.has(peerId) && peers.size < CONFIG.MAX_PEERS) {
      createPeerConnection(peerId, true);
    }
  });
}

/**
 * Handle peer disconnection notifications
 */
function handlePeerDisconnect(peerId) {
  console.log(`👋 Peer disconnected: ${peerId}`);
  cleanupPeer(peerId);
}

/**
 * Handle emergency broadcasts from server
 */
function handleEmergencyBroadcast(data) {
  console.log('🚨 Received emergency broadcast from server');
  
  if (externalCallback) {
    externalCallback({
      ...data,
      source: 'server',
      timestamp: Date.now()
    });
  }
  
  // Relay to connected peers
  relayMessage(data, 'emergency_server_relay');
}

/**
 * Create enhanced peer connection
 */
function createPeerConnection(peerId, initiator) {
  if (peers.has(peerId)) {
    console.log(`🔄 Peer ${peerId} already exists, skipping`);
    return;
  }
  
  console.log(`🤝 Creating peer connection to ${peerId} (initiator: ${initiator})`);
  
  const peer = new SimplePeer({
    initiator: initiator,
    trickle: false,
    config: {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ]
    }
  });
  
  // Store peer connection info
  const connectionInfo = {
    peerId: peerId,
    initiator: initiator,
    connected: false,
    connectedAt: null,
    lastActivity: Date.now(),
    messageCount: 0,
    reconnectAttempts: 0
  };
  
  peers.set(peerId, peer);
  peerConnections.set(peerId, connectionInfo);
  
  // Setup peer event handlers
  setupPeerEventHandlers(peer, peerId, connectionInfo);
  
  return peer;
}

/**
 * Setup event handlers for peer connections
 */
function setupPeerEventHandlers(peer, peerId, connectionInfo) {
  peer.on('signal', (data) => {
    console.log(`📤 Sending signal to ${peerId}`);
    
    if (socket && socket.connected) {
      socket.emit('signal', {
        to: peerId,
        from: localPeerId,
        data: data,
        type: connectionInfo.initiator ? 'offer' : 'answer'
      });
    }
  });
  
  peer.on('connect', () => {
    console.log(`✅ P2P connection established with ${peerId}`);
    
    connectionInfo.connected = true;
    connectionInfo.connectedAt = Date.now();
    connectionStats.totalConnections++;
    connectionStats.activeConnections++;
    
    p2pActive = true;
    updateDiagnostics(`Connected to ${peers.size} peers`, 'success');
    
    // Send introduction message
    sendToPeer(peerId, {
      type: 'introduction',
      peerId: localPeerId,
      capabilities: ['emergency_relay', 'mesh_routing'],
      timestamp: Date.now()
    });
    
    eventEmitter.emit('peer_connected', { peerId, connectionInfo });
  });
  
  peer.on('data', (data) => {
    try {
      const message = JSON.parse(data.toString());
      handlePeerMessage(peerId, message, connectionInfo);
    } catch (error) {
      console.error(`❌ Error parsing message from ${peerId}:`, error);
    }
  });
  
  peer.on('error', (error) => {
    console.error(`❌ Peer connection error with ${peerId}:`, error);
    
    // Attempt reconnection for important peers
    if (connectionInfo.reconnectAttempts < CONFIG.PEER_RECONNECT_ATTEMPTS) {
      connectionInfo.reconnectAttempts++;
      
      setTimeout(() => {
        console.log(`🔄 Attempting to reconnect to ${peerId} (attempt ${connectionInfo.reconnectAttempts})`);
        cleanupPeer(peerId);
        createPeerConnection(peerId, connectionInfo.initiator);
      }, CONFIG.PEER_RECONNECT_DELAY * connectionInfo.reconnectAttempts);
    } else {
      console.log(`❌ Max reconnection attempts reached for ${peerId}`);
      cleanupPeer(peerId);
    }
  });
  
  peer.on('close', () => {
    console.log(`👋 Peer connection closed: ${peerId}`);
    cleanupPeer(peerId);
  });
}

/**
 * Handle messages from peer connections
 */
function handlePeerMessage(peerId, message, connectionInfo) {
  connectionInfo.lastActivity = Date.now();
  connectionInfo.messageCount++;
  connectionStats.messagesReceived++;
  connectionStats.lastActivity = Date.now();
  
  console.log(`📨 Message from ${peerId}:`, message.type);
  
  switch (message.type) {
    case 'broadcast':
    case 'emergency_alert':
    case 'medical_emergency':
    case 'evacuation_order':
      // Handle emergency broadcasts
      if (externalCallback) {
        externalCallback({
          ...message.payload,
          source: 'peer',
          fromPeer: peerId,
          relayCount: (message.relayCount || 0) + 1
        });
      }
      
      // Relay to other peers (avoid loops)
      if (message.relayCount < 3) {
        relayMessage(message.payload, message.type, peerId, message.relayCount + 1);
      }
      break;
      
    case 'introduction':
      console.log(`👋 Introduction from ${peerId}:`, message);
      break;
      
    case 'heartbeat':
      // Respond to heartbeat
      sendToPeer(peerId, {
        type: 'heartbeat_response',
        timestamp: Date.now()
      });
      break;
      
    case 'heartbeat_response':
      // Update connection health
      connectionInfo.lastActivity = Date.now();
      break;
      
    default:
      console.log(`📋 Unknown message type from ${peerId}:`, message.type);
  }
  
  updateDiagnostics(`Last msg: ${message.type} @ ${new Date().toLocaleTimeString()}`);
}

/**
 * Enhanced manual pairing with better UX
 */
export async function manualSignalOffer() {
  console.log('🔗 Generating manual pairing offer');
  
  try {
    // Clean up existing manual peer
    if (manualPeer && !manualPeer.destroyed) {
      manualPeer.destroy();
    }
    
    const manualPeerId = `manual_${generatePeerId()}`;
    manualPeer = new SimplePeer({ 
      initiator: true, 
      trickle: false,
      config: {
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ]
      }
    });
    
    manualPeer.on('signal', async (data) => {
      try {
        const signalString = JSON.stringify(data);
        const qrCanvas = document.getElementById('qr-code');
        const signalOut = document.getElementById('signal-raw');
        
        // Display raw signal
        if (signalOut) {
          signalOut.textContent = signalString;
        }
        
        // Generate QR code
        if (qrCanvas && !qrRendered) {
          await QRCode.toCanvas(qrCanvas, signalString, {
            width: CONFIG.QR_CODE_SIZE,
            margin: 2,
            color: {
              dark: '#000000',
              light: '#FFFFFF'
            }
          });
          qrRendered = true;
          console.log('📱 QR code generated for manual pairing');
        }
        
      } catch (error) {
        console.error('❌ QR code generation failed:', error);
        updateDiagnostics('QR Generation Failed', 'error');
      }
    });
    
    manualPeer.on('connect', () => {
      console.log('✅ Manual P2P connection established');
      
      peers.set(manualPeerId, manualPeer);
      peerConnections.set(manualPeerId, {
        peerId: manualPeerId,
        initiator: true,
        connected: true,
        connectedAt: Date.now(),
        lastActivity: Date.now(),
        messageCount: 0,
        reconnectAttempts: 0
      });
      
      connectionStats.totalConnections++;
      connectionStats.activeConnections++;
      p2pActive = true;
      
      updateDiagnostics('Manual P2P Connected', 'success');
      
      // Send introduction
      manualPeer.send(JSON.stringify({
        type: 'introduction',
        peerId: localPeerId,
        capabilities: ['emergency_relay', 'mesh_routing'],
        timestamp: Date.now()
      }));
    });
    
    manualPeer.on('data', (data) => {
      try {
        const message = JSON.parse(data.toString());
        handlePeerMessage(manualPeerId, message, peerConnections.get(manualPeerId));
      } catch (error) {
        console.error('❌ Error parsing manual peer message:', error);
      }
    });
    
    manualPeer.on('error', (error) => {
      console.error('❌ Manual peer error:', error);
      updateDiagnostics('Manual Connection Error', 'error');
    });
    
  } catch (error) {
    console.error('❌ Manual signal offer failed:', error);
    updateDiagnostics('Manual Pairing Failed', 'error');
  }
}

/**
 * Enhanced manual signal input with validation
 */
export function manualSignalInput(input) {
  if (!input || typeof input !== 'string') {
    showUserMessage('❌ Please provide a valid signal string', 'error');
    return;
  }
  
  try {
    const data = JSON.parse(input.trim());
    
    // Validate signal structure
    if (!data.type || !data.sdp) {
      throw new Error('Invalid signal format');
    }
    
    if (manualPeer && !manualPeer.destroyed) {
      manualPeer.signal(data);
      updateDiagnostics('Manual signal accepted', 'success');
      console.log('📡 Manual signal applied successfully');
      showUserMessage('✅ Signal accepted, attempting connection...', 'success');
    } else {
      console.warn('⚠️ No manual peer instance available');
      showUserMessage('⚠️ No pairing session active. Please refresh the page.', 'warning');
    }
    
  } catch (error) {
    console.error('❌ Invalid signal input:', error);
    showUserMessage('❌ Invalid signal format. Please check the input.', 'error');
  }
}

/**
 * Enhanced broadcast with priority and reliability
 */
export function sendP2PBroadcast(payload, options = {}) {
  const messageType = options.type || 'broadcast';
  const priority = options.priority || 'normal';
  const reliability = options.reliability || 'best-effort';
  
  console.log(`📡 Broadcasting message: ${messageType} (priority: ${priority})`);
  
  const message = {
    type: messageType,
    payload: payload,
    messageId: generateMessageId(),
    timestamp: Date.now(),
    priority: priority,
    reliability: reliability,
    fromPeer: localPeerId,
    relayCount: 0
  };
  
  // Add to message queue for reliability
  if (reliability === 'guaranteed') {
    messageQueue.push({
      ...message,
      attempts: 0,
      maxAttempts: 3,
      nextAttempt: Date.now()
    });
  }
  
  // Send to all connected peers
  let sentCount = 0;
  peers.forEach((peer, peerId) => {
    if (peer.connected) {
      sendToPeer(peerId, message);
      sentCount++;
    }
  });
  
  // Send to server if available
  if (socket && socket.connected) {
    socket.emit('emergency_broadcast', message);
    sentCount++;
  }
  
  connectionStats.messagesSent++;
  
  console.log(`📤 Broadcast sent to ${sentCount} destinations`);
  updateDiagnostics(`Broadcast sent to ${sentCount} peers`);
  
  return sentCount > 0;
}

/**
 * Send message to specific peer
 */
function sendToPeer(peerId, message) {
  const peer = peers.get(peerId);
  
  if (peer && peer.connected) {
    try {
      const messageString = JSON.stringify(message);
      
      // Check message size
      if (messageString.length > CONFIG.MAX_MESSAGE_SIZE) {
        console.warn(`⚠️ Message too large for ${peerId}: ${messageString.length} bytes`);
        return false;
      }
      
      peer.send(messageString);
      
      // Update connection info
      const connectionInfo = peerConnections.get(peerId);
      if (connectionInfo) {
        connectionInfo.lastActivity = Date.now();
        connectionInfo.messageCount++;
      }
      
      connectionStats.bytesTransferred += messageString.length;
      
      return true;
    } catch (error) {
      console.error(`❌ Error sending message to ${peerId}:`, error);
      return false;
    }
  }
  
  return false;
}

/**
 * Relay message to other peers (avoid loops)
 */
function relayMessage(payload, messageType, excludePeer = null, relayCount = 0) {
  if (relayCount >= 3) {
    console.log('📡 Max relay count reached, stopping propagation');
    return;
  }
  
  console.log(`🔄 Relaying ${messageType} message (count: ${relayCount})`);
  
  peers.forEach((peer, peerId) => {
    if (peerId !== excludePeer && peer.connected) {
      sendToPeer(peerId, {
        type: messageType,
        payload: payload,
        relayCount: relayCount,
        relayedBy: localPeerId,
        timestamp: Date.now()
      });
    }
  });
}

/**
 * Setup heartbeat system for connection health
 */
function setupHeartbeat() {
  setInterval(() => {
    const now = Date.now();
    
    peers.forEach((peer, peerId) => {
      const connectionInfo = peerConnections.get(peerId);
      
      if (peer.connected && connectionInfo) {
        // Check if peer is responsive
        const timeSinceActivity = now - connectionInfo.lastActivity;
        
        if (timeSinceActivity > CONFIG.HEARTBEAT_INTERVAL * 2) {
          console.warn(`⚠️ Peer ${peerId} appears unresponsive`);
          cleanupPeer(peerId);
        } else if (timeSinceActivity > CONFIG.HEARTBEAT_INTERVAL) {
          // Send heartbeat
          sendToPeer(peerId, {
            type: 'heartbeat',
            timestamp: now
          });
        }
      }
    });
    
  }, CONFIG.HEARTBEAT_INTERVAL);
}

/**
 * Setup automatic cleanup
 */
function setupAutoCleanup() {
  setInterval(() => {
    const now = Date.now();
    
    // Clean up old messages from queue
    messageQueue = messageQueue.filter(msg => {
      return now - msg.timestamp < CONFIG.MESSAGE_TIMEOUT;
    });
    
    // Update active connections count
    connectionStats.activeConnections = Array.from(peers.values())
      .filter(peer => peer.connected).length;
    
    // Update P2P active status
    p2pActive = connectionStats.activeConnections > 0;
    
  }, CONFIG.AUTO_CLEANUP_INTERVAL);
}

/**
 * Setup statistics updates
 */
function setupStatsUpdates() {
  setInterval(() => {
    eventEmitter.emit('stats_update', connectionStats);
    
    // Update UI diagnostics
    const statusText = p2pActive 
      ? `Connected (${connectionStats.activeConnections} peers)`
      : 'Disconnected';
    
    updateDiagnostics(statusText);
    
  }, CONFIG.STATS_UPDATE_INTERVAL);
}

/**
 * Clean up peer connection
 */
function cleanupPeer(peerId) {
  const peer = peers.get(peerId);
  const connectionInfo = peerConnections.get(peerId);
  
  if (peer) {
    if (!peer.destroyed) {
      peer.destroy();
    }
    peers.delete(peerId);
  }
  
  if (connectionInfo && connectionInfo.connected) {
    connectionStats.activeConnections--;
  }
  
  peerConnections.delete(peerId);
  
  console.log(`🧹 Cleaned up peer: ${peerId}`);
  
  // Update P2P status
  p2pActive = connectionStats.activeConnections > 0;
  updateDiagnostics(p2pActive ? `Connected (${connectionStats.activeConnections} peers)` : 'Disconnected');
}

/**
 * Enhanced diagnostics and status updates
 */
function updateDiagnostics(status, type = 'info') {
  // Update various UI elements
  const statusElements = [
    'mesh-mode',
    'p2p-status',
    'connection-status'
  ];
  
  statusElements.forEach(elementId => {
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = status;
      element.className = `status-${type}`;
    }
  });
  
  // Update last message display
  const lastMsgEl = document.getElementById('last-msg');
  if (lastMsgEl) {
    lastMsgEl.textContent = status;
  }
  
  // Update connection count
  const connectionCountEl = document.getElementById('connection-count');
  if (connectionCountEl) {
    connectionCountEl.textContent = connectionStats.activeConnections;
  }
  
  // Emit status update event
  eventEmitter.emit('status_update', {
    status: status,
    type: type,
    timestamp: Date.now(),
    stats: connectionStats
  });
}

/**
 * Show user messages
 */
function showUserMessage(message, type = 'info') {
  console.log(`📢 User message (${type}): ${message}`);
  
  // Try to show in UI if elements exist
  const messageElements = document.querySelectorAll('.p2p-message, .status-message');
  messageElements.forEach(element => {
    element.textContent = message;
    element.className = `message message-${type}`;
    element.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
      element.style.display = 'none';
    }, 5000);
  });
}

/**
 * Utility functions
 */
function generatePeerId() {
  return `peer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function generateMessageId() {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Public API functions
 */
export function isP2PActive() {
  return p2pActive;
}

export function getConnectionStats() {
  return { ...connectionStats };
}

export function getConnectedPeers() {
  return Array.from(peers.keys()).filter(peerId => {
    const peer = peers.get(peerId);
    return peer && peer.connected;
  });
}

export function disconnectPeer(peerId) {
  cleanupPeer(peerId);
}

export function disconnectAll() {
  console.log('🔌 Disconnecting all P2P connections');
  
  peers.forEach((peer, peerId) => {
    cleanupPeer(peerId);
  });
  
  if (socket) {
    socket.disconnect();
  }
  
  p2pActive = false;
  updateDiagnostics('All Disconnected', 'warning');
}

export function reconnectAll() {
  console.log('🔄 Attempting to reconnect all P2P connections');
  
  if (socket && !socket.connected) {
    socket.connect();
  }
  
  // Regenerate manual pairing
  setTimeout(manualSignalOffer, 1000);
}

// Event listener access
export function onPeerConnected(callback) {
  eventEmitter.on('peer_connected', callback);
}

export function onStatusUpdate(callback) {
  eventEmitter.on('status_update', callback);
}

export function onStatsUpdate(callback) {
  eventEmitter.on('stats_update', callback);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  disconnectAll();
});

// Global functions for manual pairing UI
window.handleManualSignal = manualSignalInput;

console.log('🚀 Enhanced P2P WebRTC module loaded');