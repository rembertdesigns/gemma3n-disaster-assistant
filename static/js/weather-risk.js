// Enhanced weather-risk.js - Emergency Risk Assessment System
import { sendP2PBroadcast } from "./p2p/fallback-webrtc.js";

// Configuration
const CONFIG = {
  maxRetries: 3,
  cacheTimeout: 1800000, // 30 minutes
  updateInterval: 300000, // 5 minutes
  defaultLocation: { lat: 34.05, lon: -118.25 }, // Los Angeles default
  geolocationTimeout: 10000,
  highRiskThreshold: 0.8
};

// Global state
let isInitialized = false;
let updateTimer = null;
let lastKnownLocation = null;

/**
 * Main initialization function
 */
async function fetchRiskPrediction() {
  if (isInitialized) return;
  isInitialized = true;
  
  try {
    const location = await getCurrentLocation();
    lastKnownLocation = location;
    console.log("📍 Location acquired:", location);
    await sendPredictionRequest(location.lat, location.lon);
  } catch (error) {
    console.error("Failed to get location:", error);
    showUserError("Location services unavailable. Using default area.");
    
    // Try to load cached data first
    if (!loadCachedRiskData()) {
      // Fallback to default location
      await sendPredictionRequest(CONFIG.defaultLocation.lat, CONFIG.defaultLocation.lon);
    }
  }
}

/**
 * Enhanced geolocation with better error handling
 */
function getCurrentLocation() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation not supported'));
      return;
    }
    
    const options = {
      enableHighAccuracy: true,
      timeout: CONFIG.geolocationTimeout,
      maximumAge: 300000 // 5 minutes
    };
    
    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: Date.now()
        });
      },
      (error) => {
        console.warn('Geolocation error:', error.message);
        // Return default location instead of rejecting
        resolve({
          lat: CONFIG.defaultLocation.lat,
          lon: CONFIG.defaultLocation.lon,
          accuracy: null,
          timestamp: Date.now(),
          isDefault: true
        });
      },
      options
    );
  });
}

/**
 * Enhanced prediction request with retry logic and offline support
 */
async function sendPredictionRequest(lat, lon, retryCount = 0) {
  const weather = {
    temperature: 42,
    wind_speed: 60,
    rainfall: 20
  };
  const hazardType = "wildfire";
  
  try {
    showLoadingState(true);
    
    const response = await fetch("/predict-risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: { lat, lon },
        weather,
        hazard_type: hazardType
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    data.location = { lat, lon }; // attach location if missing
    data.timestamp = Date.now(); // add timestamp
    
    updateRiskUI(data);
    analyzeSentiment(data.severity); // 🧠 Analyze tone
    cacheRiskData(data);
    
    showLoadingState(false);
    
  } catch (error) {
    console.error("❌ Error fetching risk prediction:", error);
    
    showLoadingState(false);
    
    // Retry logic for network failures
    if (retryCount < CONFIG.maxRetries && navigator.onLine) {
      console.warn(`⚠️ Retry ${retryCount + 1}/${CONFIG.maxRetries} for risk prediction`);
      const delay = 1000 * Math.pow(2, retryCount); // exponential backoff
      setTimeout(() => sendPredictionRequest(lat, lon, retryCount + 1), delay);
      return;
    }
    
    // Try to load cached data
    if (!loadCachedRiskData()) {
      // Show error state
      updateErrorUI(error.message);
    }
    
    showUserError("Unable to fetch current risk data. " + (retryCount === 0 ? "Retrying..." : "Using cached information."));
  }
}

/**
 * Enhanced UI updates for refactored templates
 */
function updateRiskUI(data) {
  try {
    // Update risk score with proper styling
    updateRiskScore(data.risk_score, data.severity);
    
    // Update severity assessment
    updateSeverity(data.severity, data.description);
    
    // Update threats list
    updateThreats(data.threats || generateThreatsFromScore(data.risk_score));
    
    // Update resources
    updateResources(data.suggested_resources || {});
    
    // Update last updated timestamp
    updateTimestamp();
    
    // Trigger emergency response if needed
    if (data.risk_score >= CONFIG.highRiskThreshold) {
      triggerEmergencyResponse(data);
    }
    
    console.log("✅ Risk UI updated successfully");
    
  } catch (error) {
    console.error("Error updating risk UI:", error);
  }
}

/**
 * Update risk score display
 */
function updateRiskScore(score, level) {
  const scoreElement = document.getElementById('risk-score');
  const cardElement = document.getElementById('riskScoreCard');
  const descElement = document.getElementById('risk-description');
  
  if (scoreElement) {
    const percentage = Math.round(score * 100);
    scoreElement.textContent = percentage + '%';
    
    // Apply color coding based on score
    const colorClass = getScoreColorClass(score);
    scoreElement.className = `metric-value ${colorClass}`;
  }
  
  if (cardElement) {
    cardElement.className = `risk-card ${level || getLevelFromScore(score)}`;
  }
  
  if (descElement) {
    const descriptions = {
      low: "Conditions are stable with minimal immediate threats detected.",
      medium: "Moderate risk factors present. Continue monitoring conditions.",
      high: "Elevated risk conditions. Prepare for potential emergency response.",
      critical: "CRITICAL: Immediate emergency response may be required."
    };
    
    const currentLevel = level || getLevelFromScore(score);
    descElement.textContent = descriptions[currentLevel] || "Risk assessment in progress...";
  }
}

/**
 * Update severity display
 */
function updateSeverity(severity, description) {
  const severityElement = document.getElementById('severity');
  const cardElement = document.getElementById('severityCard');
  const descElement = document.getElementById('severity-description');
  
  if (severityElement) {
    severityElement.textContent = (severity || 'UNKNOWN').toUpperCase();
  }
  
  if (cardElement) {
    cardElement.className = `risk-card ${severity || 'medium'}`;
  }
  
  if (descElement) {
    descElement.textContent = description || `${severity || 'Unknown'} threat level detected.`;
  }
}

/**
 * Update threats display
 */
function updateThreats(threats) {
  const countElement = document.getElementById('threat-count');
  const listElement = document.getElementById('threats-list');
  
  if (countElement) {
    countElement.textContent = threats.length;
  }
  
  if (listElement) {
    if (threats.length > 0) {
      listElement.innerHTML = threats.map(threat => 
        `<div style="margin-bottom: 0.25rem;">• ${threat}</div>`
      ).join('');
    } else {
      listElement.textContent = "No immediate threats detected.";
    }
  }
}

/**
 * Update resources display
 */
function updateResources(resources) {
  const resourcesElement = document.getElementById('resources');
  
  if (resourcesElement) {
    if (typeof resources === 'object' && Object.keys(resources).length > 0) {
      resourcesElement.textContent = JSON.stringify(resources, null, 2);
    } else {
      resourcesElement.textContent = "No specific resource recommendations at this time.";
    }
  }
  
  // Update individual resource counts if elements exist
  const resourceTypes = ['ambulance', 'fire', 'police', 'hospital', 'air', 'shelter'];
  resourceTypes.forEach(type => {
    const element = document.getElementById(`${type}-count`);
    if (element && resources[type + 's']) {
      element.textContent = resources[type + 's'];
    }
  });
}

/**
 * Update timestamp display
 */
function updateTimestamp() {
  const timestampElements = document.querySelectorAll('.last-updated, #lastUpdated');
  const now = new Date().toLocaleTimeString();
  
  timestampElements.forEach(element => {
    element.textContent = `Last updated: ${now}`;
  });
}

/**
 * Show error state in UI
 */
function updateErrorUI(errorMessage) {
  const scoreElement = document.getElementById("risk-score");
  const severityElement = document.getElementById("severity");
  const resourcesElement = document.getElementById("resources");
  
  if (scoreElement) scoreElement.textContent = "Error";
  if (severityElement) severityElement.textContent = "Error";
  if (resourcesElement) resourcesElement.textContent = `Failed to load: ${errorMessage}`;
}

/**
 * Enhanced emergency response trigger
 */
async function triggerEmergencyResponse(data) {
  console.log("🚨 Triggering emergency response for high risk situation");
  
  // Trigger broadcast
  await triggerBroadcast(data);
  
  // Send P2P broadcast
  const broadcastPayload = {
    message: `⚠️ High disaster risk (${data.severity}) detected at your location!`,
    severity: data.severity,
    location: data.location,
    timestamp: Date.now(),
    type: 'emergency_alert'
  };
  
  try {
    await sendP2PBroadcast(broadcastPayload);
    console.log("📡 P2P emergency broadcast sent");
  } catch (error) {
    console.error("Failed to send P2P broadcast:", error);
  }
  
  // Show emergency alert banner
  showEmergencyAlert(data);
}

/**
 * Enhanced broadcast trigger
 */
async function triggerBroadcast(data) {
  const payload = {
    message: `🚨 Emergency Risk Detected (${data.severity})`,
    location: data.location,
    severity: data.severity,
    risk_score: data.risk_score,
    timestamp: Date.now()
  };
  
  try {
    const response = await fetch("/broadcast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`Broadcast failed: ${response.status}`);
    }
    
    const result = await response.json();
    console.log("📢 Broadcast triggered successfully:", result);
    
  } catch (error) {
    console.error("❌ Failed to trigger broadcast:", error);
    
    // Store for offline sync
    const syncQueue = JSON.parse(localStorage.getItem('syncQueue') || '[]');
    syncQueue.push({
      id: Date.now().toString(),
      type: 'emergency_broadcast',
      data: payload,
      timestamp: new Date().toISOString()
    });
    localStorage.setItem('syncQueue', JSON.stringify(syncQueue));
  }
}

/**
 * Show emergency alert banner
 */
function showEmergencyAlert(data) {
  const alertBanner = document.getElementById("broadcast-alert");
  if (alertBanner) {
    alertBanner.style.display = "block";
    alertBanner.classList.add("active");
    alertBanner.innerHTML = `
      🚨 <strong>EMERGENCY ALERT:</strong> ${data.severity.toUpperCase()} risk detected near your location. 
      Risk Score: ${Math.round(data.risk_score * 100)}%
    `;
    
    // Auto-hide after 10 seconds
    setTimeout(() => {
      alertBanner.classList.remove("active");
    }, 10000);
  }
}

/**
 * Enhanced sentiment analysis with better error handling
 */
async function analyzeSentiment(text) {
  try {
    const response = await fetch("/analyze-sentiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text || 'unknown' })
    });
    
    if (!response.ok) {
      throw new Error(`Sentiment analysis failed: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Update UI elements
    updateSentimentUI(result.sentiment, result.escalation, result.confidence);
    
  } catch (error) {
    console.warn("⚠️ Sentiment analysis failed:", error);
    
    // Fallback sentiment based on severity
    const fallbackSentiment = getFallbackSentiment(text);
    updateSentimentUI(fallbackSentiment.sentiment, fallbackSentiment.escalation, fallbackSentiment.confidence);
  }
}

/**
 * Update sentiment UI elements
 */
function updateSentimentUI(sentiment, escalation, confidence) {
  const sentimentElement = document.getElementById("sentiment-result");
  const escalationElement = document.getElementById("escalation-level");
  const confidenceElement = document.getElementById("sentiment-confidence");
  
  if (sentimentElement) {
    sentimentElement.innerHTML = `<strong>${sentiment || 'Unknown'}</strong>`;
    sentimentElement.className = `sentiment-score sentiment-${escalation || 'neutral'}`;
  }
  
  if (escalationElement) {
    escalationElement.innerHTML = `<strong>${escalation || 'Unknown'}</strong>`;
    escalationElement.className = `sentiment-score sentiment-${escalation || 'neutral'}`;
  }
  
  if (confidenceElement) {
    confidenceElement.innerHTML = `<strong>${confidence || 'N/A'}%</strong>`;
  }
}

/**
 * Cache management functions
 */
function cacheRiskData(data) {
  try {
    const cacheData = {
      data: data,
      timestamp: Date.now(),
      location: lastKnownLocation
    };
    localStorage.setItem('lastRiskData', JSON.stringify(cacheData));
    console.log("💾 Risk data cached successfully");
  } catch (error) {
    console.warn('Failed to cache risk data:', error);
  }
}

function loadCachedRiskData() {
  try {
    const cached = localStorage.getItem('lastRiskData');
    if (!cached) return false;
    
    const { data, timestamp, location } = JSON.parse(cached);
    
    // Check if cache is still valid (30 minutes)
    if (Date.now() - timestamp > CONFIG.cacheTimeout) {
      console.log("📅 Cached data expired");
      return false;
    }
    
    console.log("💾 Loading cached risk data");
    updateRiskUI(data);
    showUserMessage('Showing cached risk data (offline mode)', 'warning');
    
    return true;
    
  } catch (error) {
    console.warn('Failed to load cached data:', error);
    return false;
  }
}

/**
 * Loading state management
 */
function showLoadingState(isLoading) {
  const loadingElements = document.querySelectorAll('.loading, .loading-spinner, #loadingIndicator');
  const scoreElement = document.getElementById('risk-score');
  const severityElement = document.getElementById('severity');
  
  if (isLoading) {
    loadingElements.forEach(el => {
      if (el) el.style.display = 'block';
    });
    
    if (scoreElement) scoreElement.innerHTML = '<span class="loading"></span>';
    if (severityElement) severityElement.innerHTML = '<span class="loading"></span>';
  } else {
    loadingElements.forEach(el => {
      if (el) el.style.display = 'none';
    });
  }
}

/**
 * User notification system
 */
function showUserError(message) {
  showNotification(message, 'error');
}

function showUserMessage(message, type = 'info') {
  showNotification(message, type);
}

function showNotification(message, type) {
  // Remove existing notifications
  document.querySelectorAll('.risk-notification').forEach(el => el.remove());
  
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `risk-notification notification-${type}`;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    color: white;
    font-weight: 500;
    z-index: 9999;
    max-width: 350px;
    font-size: 0.9rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    animation: slideIn 0.3s ease;
    ${type === 'error' ? 'background: #dc2626;' : 
      type === 'warning' ? 'background: #f59e0b;' : 
      type === 'success' ? 'background: #16a34a;' :
      'background: #3b82f6;'}
  `;
  notification.innerHTML = `
    <div style="display: flex; align-items: center; gap: 0.5rem;">
      <span>${type === 'error' ? '⚠️' : type === 'warning' ? '📢' : type === 'success' ? '✅' : 'ℹ️'}</span>
      <span>${message}</span>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  // Auto-remove after 5 seconds
  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => notification.remove(), 300);
  }, 5000);
}

/**
 * Utility functions
 */
function getScoreColorClass(score) {
  if (score >= 0.8) return 'risk-score-76-100';
  if (score >= 0.6) return 'risk-score-51-75';
  if (score >= 0.3) return 'risk-score-26-50';
  return 'risk-score-0-25';
}

function getLevelFromScore(score) {
  if (score >= 0.8) return 'critical';
  if (score >= 0.6) return 'high';
  if (score >= 0.3) return 'medium';
  return 'low';
}

function generateThreatsFromScore(score) {
  const threats = [];
  if (score >= 0.7) threats.push('Extreme Weather Conditions');
  if (score >= 0.6) threats.push('Infrastructure Vulnerability');
  if (score >= 0.5) threats.push('High Population Density');
  if (score >= 0.4) threats.push('Limited Evacuation Routes');
  if (score >= 0.3) threats.push('Resource Constraints');
  
  return threats.length > 0 ? threats : ['No specific threats identified'];
}

function getFallbackSentiment(text) {
  const severity = (text || '').toLowerCase();
  
  if (severity.includes('critical') || severity.includes('emergency')) {
    return { sentiment: 'Critical', escalation: 'critical', confidence: 75 };
  } else if (severity.includes('high') || severity.includes('severe')) {
    return { sentiment: 'Concerned', escalation: 'elevated', confidence: 70 };
  } else if (severity.includes('medium') || severity.includes('moderate')) {
    return { sentiment: 'Cautious', escalation: 'concerned', confidence: 65 };
  } else {
    return { sentiment: 'Calm', escalation: 'calm', confidence: 60 };
  }
}

/**
 * Page-specific initialization
 */
function initializeForPage() {
  const currentPage = window.location.pathname;
  
  console.log("🚀 Initializing weather risk system for:", currentPage);
  
  if (currentPage.includes('predict')) {
    // Full dashboard initialization
    fetchRiskPrediction();
    startPeriodicUpdates();
    console.log("📊 Full risk dashboard initialized");
  } else if (currentPage.includes('submit-report') || currentPage.includes('crowd-report')) {
    // Background risk monitoring
    fetchRiskPrediction();
    console.log("📝 Background risk monitoring initialized");
  } else {
    // Basic initialization for other pages
    console.log("📍 Basic risk monitoring initialized");
  }
  
  // Add online/offline event listeners
  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);
}

/**
 * Periodic updates management
 */
function startPeriodicUpdates() {
  if (updateTimer) {
    clearInterval(updateTimer);
  }
  
  updateTimer = setInterval(() => {
    if (navigator.onLine && document.visibilityState === 'visible') {
      console.log("🔄 Periodic risk update");
      if (lastKnownLocation) {
        sendPredictionRequest(lastKnownLocation.lat, lastKnownLocation.lon);
      } else {
        fetchRiskPrediction();
      }
    }
  }, CONFIG.updateInterval);
  
  console.log("⏰ Periodic updates started (5 minute interval)");
}

function stopPeriodicUpdates() {
  if (updateTimer) {
    clearInterval(updateTimer);
    updateTimer = null;
    console.log("⏰ Periodic updates stopped");
  }
}

/**
 * Online/Offline event handlers
 */
function handleOnline() {
  console.log("🌐 Connection restored");
  showUserMessage("Connection restored - updating risk data", "success");
  
  // Immediately fetch fresh data
  if (lastKnownLocation) {
    sendPredictionRequest(lastKnownLocation.lat, lastKnownLocation.lon);
  } else {
    fetchRiskPrediction();
  }
  
  // Restart periodic updates
  if (window.location.pathname.includes('predict')) {
    startPeriodicUpdates();
  }
}

function handleOffline() {
  console.log("📴 Connection lost");
  showUserMessage("Connection lost - using cached data", "warning");
  
  // Stop periodic updates
  stopPeriodicUpdates();
  
  // Try to load cached data
  loadCachedRiskData();
}

/**
 * Cleanup function
 */
function cleanup() {
  stopPeriodicUpdates();
  window.removeEventListener('online', handleOnline);
  window.removeEventListener('offline', handleOffline);
  console.log("🧹 Weather risk system cleanup completed");
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  @keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
  }
  .loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", initializeForPage);

// Cleanup on page unload
window.addEventListener('beforeunload', cleanup);

// Export functions for external use
export {
  fetchRiskPrediction,
  analyzeSentiment,
  triggerBroadcast,
  showUserMessage,
  showUserError,
  loadCachedRiskData,
  cacheRiskData
}; 