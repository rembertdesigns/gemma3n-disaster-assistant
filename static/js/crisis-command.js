/**
 * Crisis Command Center - Advanced Multi-Agency Coordination System
 * Real-time emergency management and resource coordination
 */

class CrisisCommandCenter {
    constructor() {
      this.isInitialized = false;
      this.map = null;
      this.incidents = [];
      this.resources = {};
      this.agencies = {};
      this.communications = {
        radio: [],
        dispatch: [],
        'inter-agency': []
      };
      this.currentCommsTab = 'radio';
      this.alertLevel = 3;
      this.operationsTimeline = [];
      
      // Resource tracking
      this.resourceTypes = {
        fireUnits: { total: 12, available: 4, deployed: 8 },
        ambulances: { total: 18, available: 11, deployed: 7 },
        policeUnits: { total: 24, available: 9, deployed: 15 },
        hazmatTeams: { total: 4, available: 2, deployed: 2 },
        helicopters: { total: 3, available: 3, deployed: 0 },
        searchRescue: { total: 6, available: 4, deployed: 2 }
      };
      
      // Agency status tracking
      this.agencyStatus = {
        'Fire Department': { status: 'active', ic: 'Battalion Chief Johnson', channel: 'Channel 1', units: 12 },
        'Police Department': { status: 'active', ic: 'Lieutenant Davis', channel: 'Channel 3', units: 15 },
        'EMS Services': { status: 'active', ic: 'Paramedic Supervisor', channel: 'Channel 2', units: 7 },
        'Public Works': { status: 'standby', ic: 'Operations Center', channel: 'Channel 5', units: 2 },
        'Red Cross': { status: 'active', ic: 'Emergency Coordinator', contact: '(555) 123-4567' },
        'Utility Company': { status: 'active', ic: 'Emergency Response', contact: '(555) 987-6543' }
      };
      
      // Incident data
      this.activeIncidents = [
        {
          id: 'INC-001',
          title: 'Structure Fire - 425 Oak Street',
          time: '14:32',
          description: 'Multi-story residential building. 15 units dispatched. Evacuation in progress.',
          priority: 'critical',
          status: 'onscene',
          location: { lat: 37.7749, lng: -122.4194 },
          assignedUnits: ['Engine 15', 'Engine 7', 'Ladder 3', 'Battalion Chief 2']
        },
        {
          id: 'INC-002',
          title: 'MVA with Injuries - Highway 101',
          time: '14:28',
          description: 'Multi-vehicle accident, northbound lanes blocked. 3 ambulances dispatched.',
          priority: 'high',
          status: 'enroute',
          location: { lat: 37.7849, lng: -122.4094 },
          assignedUnits: ['Ambulance 12', 'Ambulance 8', 'Police Unit 23']
        },
        {
          id: 'INC-003',
          title: 'Gas Leak - Downtown Commercial',
          time: '14:15',
          description: 'Natural gas leak reported. Hazmat team activated. Area evacuation ordered.',
          priority: 'high',
          status: 'onscene',
          location: { lat: 37.7649, lng: -122.4294 },
          assignedUnits: ['Hazmat 1', 'Engine 5', 'Police Unit 15']
        },
        {
          id: 'INC-004',
          title: 'Medical Emergency - Senior Center',
          time: '14:05',
          description: 'Cardiac event reported. ALS unit dispatched. Patient stable.',
          priority: 'medium',
          status: 'resolved',
          location: { lat: 37.7549, lng: -122.4394 },
          assignedUnits: ['Ambulance 5']
        }
      ];
      
      this.updateInterval = null;
      this.commsUpdateInterval = null;
    }
  
    /**
     * Initialize the Crisis Command Center
     */
    async initialize() {
      if (this.isInitialized) return;
      
      try {
        console.log('🚨 Initializing Crisis Command Center...');
        
        // Initialize map
        await this.initializeMap();
        
        // Setup real-time updates
        this.startRealTimeUpdates();
        
        // Initialize communications
        this.initializeCommunications();
        
        // Load incident markers
        this.loadIncidentMarkers();
        
        // Start communications simulation
        this.startCommunicationsSimulation();
        
        // Update displays
        this.updateIncidentDisplay();
        this.updateResourceDisplay();
        this.updateTimelineDisplay();
        
        this.isInitialized = true;
        console.log('✅ Crisis Command Center initialized successfully');
        
      } catch (error) {
        console.error('❌ Crisis Command Center initialization failed:', error);
      }
    }
  
    /**
     * Initialize the situation map
     */
    async initializeMap() {
      const mapContainer = document.getElementById('situationMap');
      if (!mapContainer) return;
      
      // Initialize Leaflet map
      this.map = L.map('situationMap').setView([37.7749, -122.4194], 13);
      
      // Add tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
      }).addTo(this.map);
      
      // Create marker groups
      this.incidentMarkers = L.layerGroup().addTo(this.map);
      this.resourceMarkers = L.layerGroup().addTo(this.map);
      
      console.log('🗺️ Situation map initialized');
    }
  
    /**
     * Load incident markers on the map
     */
    loadIncidentMarkers() {
      if (!this.map || !this.incidentMarkers) return;
      
      this.incidentMarkers.clearLayers();
      
      this.activeIncidents.forEach(incident => {
        if (incident.status !== 'resolved') {
          const marker = this.createIncidentMarker(incident);
          this.incidentMarkers.addLayer(marker);
        }
      });
    }
  
    /**
     * Create an incident marker
     */
    createIncidentMarker(incident) {
      const icon = this.getIncidentIcon(incident.priority);
      const marker = L.marker([incident.location.lat, incident.location.lng], { icon });
      
      const popupContent = `
        <div style="min-width: 200px;">
          <h4 style="margin: 0 0 0.5rem 0; color: #dc2626;">${incident.title}</h4>
          <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">${incident.description}</p>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="background: #dc2626; color: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">
              ${incident.priority.toUpperCase()}
            </span>
            <span style="font-size: 0.8rem; color: #6b7280;">
              ${incident.time}
            </span>
          </div>
          <div style="margin-top: 0.5rem; font-size: 0.8rem;">
            <strong>Assigned Units:</strong><br>
            ${incident.assignedUnits.join(', ')}
          </div>
        </div>
      `;
      
      marker.bindPopup(popupContent);
      return marker;
    }
  
    /**
     * Get icon for incident based on priority
     */
    getIncidentIcon(priority) {
      const colors = {
        critical: '#dc2626',
        high: '#ea580c',
        medium: '#eab308',
        low: '#16a34a'
      };
      
      return L.divIcon({
        className: 'custom-incident-marker',
        html: `<div style="
          background: ${colors[priority] || colors.medium};
          width: 20px;
          height: 20px;
          border-radius: 50%;
          border: 3px solid white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
          animation: pulse 2s infinite;
        "></div>`,
        iconSize: [26, 26],
        iconAnchor: [13, 13]
      });
    }
  
    /**
     * Start real-time updates
     */
    startRealTimeUpdates() {
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
      }
      
      this.updateInterval = setInterval(() => {
        this.simulateUpdates();
        this.updateDisplays();
      }, 5000); // Update every 5 seconds
      
      console.log('🔄 Real-time updates started');
    }
  
    /**
     * Simulate real-time updates
     */
    simulateUpdates() {
      // Simulate resource changes
      Object.keys(this.resourceTypes).forEach(type => {
        const resource = this.resourceTypes[type];
        
        // Randomly deploy/return resources
        if (Math.random() < 0.1) { // 10% chance of change
          const change = Math.random() < 0.5 ? 1 : -1;
          const newDeployed = Math.max(0, Math.min(resource.total, resource.deployed + change));
          resource.deployed = newDeployed;
          resource.available = resource.total - resource.deployed;
        }
      });
      
      // Simulate incident status changes
      this.activeIncidents.forEach(incident => {
        if (Math.random() < 0.05 && incident.status !== 'resolved') { // 5% chance
          const statuses = ['pending', 'dispatched', 'enroute', 'onscene'];
          const currentIndex = statuses.indexOf(incident.status);
          if (currentIndex < statuses.length - 1) {
            incident.status = statuses[currentIndex + 1];
            this.addTimelineEvent(`Incident ${incident.id} status updated: ${incident.status}`);
          }
        }
      });
    }
  
    /**
     * Update all displays
     */
    updateDisplays() {
      this.updateResourceDisplay();
      this.updateIncidentDisplay();
      this.updateActiveIncidentCount();
    }
  
    /**
     * Update resource display
     */
    updateResourceDisplay() {
      Object.keys(this.resourceTypes).forEach(type => {
        const resource = this.resourceTypes[type];
        const countElement = document.getElementById(type);
        
        if (countElement) {
          countElement.textContent = resource.total;
        }
        
        // Update status in resource cards
        const cards = document.querySelectorAll('.resource-card');
        cards.forEach(card => {
          const label = card.querySelector('.resource-label');
          if (label && label.textContent.toLowerCase().includes(type.toLowerCase().replace(/([A-Z])/g, ' $1').trim())) {
            const statusElement = card.querySelector('.resource-status');
            if (statusElement) {
              if (resource.available > resource.deployed) {
                statusElement.textContent = `${resource.available} Available`;
                statusElement.className = 'resource-status status-available';
              } else if (resource.deployed > 0) {
                statusElement.textContent = `${resource.deployed} Deployed`;
                statusElement.className = 'resource-status status-deployed';
              } else {
                statusElement.textContent = 'Unavailable';
                statusElement.className = 'resource-status status-unavailable';
              }
            }
          }
        });
      });
    }
  
    /**
     * Update incident display
     */
    updateIncidentDisplay() {
      const incidentList = document.getElementById('incidentList');
      if (!incidentList) return;
      
      const incidentHTML = this.activeIncidents.map(incident => `
        <div class="incident-item">
          <div class="incident-header">
            <div class="incident-title">${incident.title}</div>
            <div class="incident-time">${incident.time}</div>
          </div>
          <div class="incident-details">
            ${incident.description}
          </div>
          <div class="incident-status">
            <span class="priority-badge priority-${incident.priority}">${incident.priority.toUpperCase()}</span>
            <span class="response-status status-${incident.status}">${incident.status.toUpperCase().replace(/([A-Z])/g, ' $1').trim()}</span>
          </div>
        </div>
      `).join('');
      
      incidentList.innerHTML = incidentHTML;
      
      // Reload incident markers
      this.loadIncidentMarkers();
    }
  
    /**
     * Update active incident count
     */
    updateActiveIncidentCount() {
      const activeCount = this.activeIncidents.filter(inc => inc.status !== 'resolved').length;
      const countElement = document.getElementById('activeIncidents');
      if (countElement) {
        countElement.textContent = activeCount;
      }
    }
  
    /**
     * Initialize communications system
     */
    initializeCommunications() {
      // Initialize with sample messages
      this.communications.radio = [
        {
          sender: 'Engine 15',
          time: '14:35:12',
          message: 'Command, Engine 15 on scene. Working structure fire, requesting additional engine company and ladder truck.'
        },
        {
          sender: 'Dispatch',
          time: '14:35:20',
          message: 'Engine 7 and Ladder 3 dispatched to your location, ETA 4 minutes.'
        },
        {
          sender: 'Battalion Chief 2',
          time: '14:33:45',
          message: 'Establishing command at Oak Street incident. Requesting Red Cross for displaced residents.'
        }
      ];
      
      this.communications.dispatch = [
        {
          sender: 'Dispatch Center',
          time: '14:34:15',
          message: 'All units, be advised: Structure fire at 425 Oak Street, multiple agencies responding.'
        },
        {
          sender: 'Unit 23',
          time: '14:30:22',
          message: 'Dispatch, Unit 23 arriving at Highway 101 MVA, requesting traffic control.'
        }
      ];
      
      this.communications['inter-agency'] = [
        {
          sender: 'Fire IC',
          time: '14:32:30',
          message: 'Red Cross needed for resident displacement at Oak Street fire.'
        },
        {
          sender: 'Police Commander',
          time: '14:31:15',
          message: 'PD establishing perimeter around gas leak downtown, requesting utility company response.'
        }
      ];
      
      this.updateCommunicationsDisplay();
    }
  
    /**
     * Start communications simulation
     */
    startCommunicationsSimulation() {
      this.commsUpdateInterval = setInterval(() => {
        this.simulateNewMessage();
      }, 15000); // New message every 15 seconds
    }
  
    /**
     * Simulate new communication message
     */
    simulateNewMessage() {
      const sampleMessages = [
        { sender: 'Engine 12', message: 'Command, Engine 12 available for assignment.' },
        { sender: 'Ambulance 8', message: 'Transport complete, Ambulance 8 available.' },
        { sender: 'Dispatch', message: 'All units, weather update: winds increasing to 15-20 mph.' },
        { sender: 'Battalion Chief 3', message: 'Command, requesting additional resources for extended operations.' },
        { sender: 'Hazmat 1', message: 'Gas leak secured, area safe for normal operations.' },
        { sender: 'Police Unit 15', message: 'Traffic control established, scene secure.' }
      ];
      
      const message = sampleMessages[Math.floor(Math.random() * sampleMessages.length)];
      const now = new Date();
      const timeString = now.toTimeString().slice(0, 8);
      
      this.communications[this.currentCommsTab].unshift({
        sender: message.sender,
        time: timeString,
        message: message.message
      });
      
      // Keep only last 20 messages per channel
      if (this.communications[this.currentCommsTab].length > 20) {
        this.communications[this.currentCommsTab] = this.communications[this.currentCommsTab].slice(0, 20);
      }
      
      this.updateCommunicationsDisplay();
    }
  
    /**
     * Switch communications tab
     */
    switchCommsTab(tab) {
      this.currentCommsTab = tab;
      
      // Update tab buttons
      const tabs = document.querySelectorAll('.comms-tab');
      tabs.forEach(tabBtn => {
        tabBtn.classList.remove('active');
        if (tabBtn.textContent.toLowerCase().includes(tab.replace('-', ' '))) {
          tabBtn.classList.add('active');
        }
      });
      
      this.updateCommunicationsDisplay();
    }
  
    /**
     * Update communications display
     */
    updateCommunicationsDisplay() {
      const commsContent = document.getElementById('commsContent');
      if (!commsContent) return;
      
      const messages = this.communications[this.currentCommsTab] || [];
      
      const messagesHTML = messages.map(msg => `
        <div class="message-item">
          <div class="message-header">
            <div class="message-sender">${msg.sender}</div>
            <div class="message-time">${msg.time}</div>
          </div>
          <div class="message-content">"${msg.message}"</div>
        </div>
      `).join('');
      
      commsContent.innerHTML = messagesHTML;
      
      // Auto-scroll to top for newest messages
      commsContent.scrollTop = 0;
    }
  
    /**
     * Update timeline display
     */
    updateTimelineDisplay() {
      const timeline = document.getElementById('operationsTimeline');
      if (!timeline) return;
      
      const timelineHTML = this.operationsTimeline.slice(0, 6).map(event => `
        <div class="timeline-item">
          <div class="timeline-time">${event.time}</div>
          <div class="timeline-title">${event.title}</div>
          <div class="timeline-description">${event.description}</div>
        </div>
      `).join('');
      
      timeline.innerHTML = timelineHTML;
    }
  
    /**
     * Add timeline event
     */
    addTimelineEvent(title, description = '') {
      const now = new Date();
      const timeString = now.toTimeString().slice(0, 5);
      
      this.operationsTimeline.unshift({
        time: timeString,
        title: title,
        description: description
      });
      
      // Keep only last 10 events
      if (this.operationsTimeline.length > 10) {
        this.operationsTimeline = this.operationsTimeline.slice(0, 10);
      }
      
      this.updateTimelineDisplay();
    }
  
    /**
     * Center map on incidents
     */
    centerMap() {
      if (!this.map || this.activeIncidents.length === 0) return;
      
      const activeIncidents = this.activeIncidents.filter(inc => inc.status !== 'resolved');
      if (activeIncidents.length === 0) return;
      
      if (activeIncidents.length === 1) {
        const incident = activeIncidents[0];
        this.map.setView([incident.location.lat, incident.location.lng], 15);
      } else {
        const group = new L.featureGroup(this.incidentMarkers.getLayers());
        this.map.fitBounds(group.getBounds().pad(0.1));
      }
      
      console.log('🎯 Map centered on active incidents');
    }
  
    /**
     * Toggle map layers
     */
    toggleLayers() {
      if (!this.map) return;
      
      if (this.map.hasLayer(this.incidentMarkers)) {
        this.map.removeLayer(this.incidentMarkers);
        console.log('📍 Incident markers hidden');
      } else {
        this.map.addLayer(this.incidentMarkers);
        console.log('📍 Incident markers shown');
      }
    }
  
    /**
     * Add marker to map
     */
    addMarker() {
      if (!this.map) return;
      
      const center = this.map.getCenter();
      const marker = L.marker([center.lat, center.lng])
        .addTo(this.map)
        .bindPopup('New marker added at current map center')
        .openPopup();
      
      console.log('📌 New marker added to map');
    }
  
    /**
     * Declare emergency
     */
    declareEmergency() {
      if (this.alertLevel < 5) {
        this.alertLevel++;
        this.updateAlertLevel();
        this.addTimelineEvent('Alert Level Increased', `Alert level raised to ${this.alertLevel}`);
        console.log(`🚨 Emergency declared - Alert Level ${this.alertLevel}`);
      }
    }
  
    /**
     * Quick dispatch
     */
    quickDispatch() {
      const availableUnits = [];
      
      Object.keys(this.resourceTypes).forEach(type => {
        const resource = this.resourceTypes[type];
        if (resource.available > 0) {
          availableUnits.push(`${type}: ${resource.available} available`);
        }
      });
      
      if (availableUnits.length > 0) {
        alert(`Available units for dispatch:\n\n${availableUnits.join('\n')}`);
        console.log('📞 Quick dispatch panel opened');
      } else {
        alert('No units currently available for dispatch.');
      }
    }
  
    /**
     * Broadcast alert
     */
    broadcastAlert() {
      const message = prompt('Enter alert message to broadcast to all agencies:');
      if (message) {
        const now = new Date();
        const timeString = now.toTimeString().slice(0, 8);
        
        // Add to all communication channels
        Object.keys(this.communications).forEach(channel => {
          this.communications[channel].unshift({
            sender: 'COMMAND CENTER',
            time: timeString,
            message: `BROADCAST ALERT: ${message}`
          });
        });
        
        this.updateCommunicationsDisplay();
        this.addTimelineEvent('Broadcast Alert Sent', message);
        console.log('📢 Alert broadcast to all agencies');
      }
    }
  
    /**
     * Check resource status
     */
    checkResources() {
      const resourceStatus = Object.keys(this.resourceTypes).map(type => {
        const resource = this.resourceTypes[type];
        const utilization = Math.round((resource.deployed / resource.total) * 100);
        return `${type}: ${utilization}% utilized (${resource.deployed}/${resource.total})`;
      }).join('\n');
      
      alert(`Current Resource Utilization:\n\n${resourceStatus}`);
      console.log('📊 Resource status check completed');
    }
  
    /**
     * Update alert level display
     */
    updateAlertLevel() {
      const alertElement = document.getElementById('alertLevel');
      if (alertElement) {
        alertElement.className = `alert-level alert-level-${this.alertLevel}`;
        alertElement.innerHTML = `
          <span>⚠️</span>
          <span>ALERT LEVEL ${this.alertLevel}</span>
        `;
      }
    }
  
    /**
     * Cleanup when page unloads
     */
    cleanup() {
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
      }
      
      if (this.commsUpdateInterval) {
        clearInterval(this.commsUpdateInterval);
      }
      
      if (this.map) {
        this.map.remove();
      }
    }
  }
  
  // Initialize global CrisisCommand instance
  window.CrisisCommand = new CrisisCommandCenter();
  
  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    if (window.CrisisCommand) {
      window.CrisisCommand.cleanup();
    }
  });
  
  // Auto-initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      window.CrisisCommand.initialize();
    });
  } else {
    window.CrisisCommand.initialize();
  }