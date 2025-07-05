// =====================================
// FILE: static/js/report-generator.js
// =====================================

class EmergencyReportGenerator {
    constructor() {
      this.reportData = {
        location: "",
        coordinates: [30.2672, -97.7431], // Default to Austin, TX
        hazards: [],
        severity: 5,
        notes: "",
        image_url: "",
        checklist: [],
        timestamp: new Date().toISOString(),
        report_id: this.generateReportId()
      };
  
      this.map = null;
      this.marker = null;
      
      this.init();
    }
  
    // =====================================
    // INITIALIZATION
    // =====================================
    
    init() {
      console.log('📋 Emergency Report Generator initialized');
      
      this.initMap();
      this.addEventListeners();
      this.addDraftControls();
      this.enableAutoSave();
      this.checkForDraft();
      this.renderPreview();
      
      this.showToast('📋 Professional report generator ready!');
    }
  
    generateReportId() {
      return 'RPT-' + new Date().getFullYear() + '-' + 
             Math.random().toString(36).substr(2, 9).toUpperCase();
    }
  
    // =====================================
    // MAP FUNCTIONALITY
    // =====================================
    
    initMap(lat = 30.2672, lon = -97.7431) {
      if (this.map) {
        this.map.remove();
      }
      
      this.map = L.map('map').setView([lat, lon], 13);
      
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
      }).addTo(this.map);
  
      this.map.on('click', (e) => {
        this.setMapMarker(e.latlng.lat, e.latlng.lng);
        this.reverseGeocode(e.latlng.lat, e.latlng.lng);
      });
  
      this.setMapMarker(lat, lon);
    }
  
    setMapMarker(lat, lon) {
      if (this.marker) {
        this.map.removeLayer(this.marker);
      }
      
      this.marker = L.marker([lat, lon]).addTo(this.map)
        .bindPopup('📍 Incident Location')
        .openPopup();
      
      this.reportData.coordinates = [lat, lon];
      this.renderPreview();
    }
  
    async reverseGeocode(lat, lon) {
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=18&addressdetails=1`,
          { headers: { 'User-Agent': 'EmergencyReportGenerator/1.0' } }
        );
        
        if (response.ok) {
          const data = await response.json();
          const address = data.display_name || `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
          document.getElementById('location').value = address;
          this.reportData.location = address;
          this.renderPreview();
        }
      } catch (error) {
        console.warn('Reverse geocoding failed:', error);
      }
    }
  
    // =====================================
    // MAP CONTROLS
    // =====================================
    
    getCurrentLocation() {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            this.map.setView([lat, lon], 15);
            this.setMapMarker(lat, lon);
            this.reverseGeocode(lat, lon);
            this.showToast('📍 Location updated to current position');
          },
          (error) => {
            this.showToast('❌ Unable to get current location', 'error');
          }
        );
      } else {
        this.showToast('❌ Geolocation not supported', 'error');
      }
    }
  
    async searchLocation() {
      const query = prompt('Enter location to search:');
      if (!query) return;
  
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`,
          { headers: { 'User-Agent': 'EmergencyReportGenerator/1.0' } }
        );
        
        if (response.ok) {
          const data = await response.json();
          if (data.length > 0) {
            const result = data[0];
            const lat = parseFloat(result.lat);
            const lon = parseFloat(result.lon);
            
            this.map.setView([lat, lon], 15);
            this.setMapMarker(lat, lon);
            document.getElementById('location').value = result.display_name;
            this.reportData.location = result.display_name;
            this.renderPreview();
            this.showToast('🔍 Location found and updated');
          } else {
            this.showToast('❌ Location not found', 'error');
          }
        }
      } catch (error) {
        this.showToast('❌ Search failed', 'error');
      }
    }
  
    resetMap() {
      this.map.setView([30.2672, -97.7431], 13);
      this.setMapMarker(30.2672, -97.7431);
      document.getElementById('location').value = '';
      this.reportData.location = '';
      this.reportData.coordinates = [30.2672, -97.7431];
      this.renderPreview();
      this.showToast('🔄 Map reset to default location');
    }
  
    // =====================================
    // CHECKLIST MANAGEMENT
    // =====================================
    
    addChecklistItem() {
      const input = document.getElementById('newTaskInput');
      const task = input.value.trim();
      
      if (task && !this.reportData.checklist.includes(task)) {
        this.reportData.checklist.push(task);
        input.value = '';
        this.updateChecklistDisplay();
        this.renderPreview();
        this.showToast('✅ Task added to checklist');
      } else if (this.reportData.checklist.includes(task)) {
        this.showToast('⚠️ Task already exists', 'warning');
      }
    }
  
    addQuickTask(task) {
      if (!this.reportData.checklist.includes(task)) {
        this.reportData.checklist.push(task);
        this.updateChecklistDisplay();
        this.renderPreview();
        this.showToast(`✅ "${task}" added to checklist`);
      } else {
        this.showToast('⚠️ Task already exists', 'warning');
      }
    }
  
    removeChecklistItem(index) {
      const removedTask = this.reportData.checklist[index];
      this.reportData.checklist.splice(index, 1);
      this.updateChecklistDisplay();
      this.renderPreview();
      this.showToast(`🗑️ "${removedTask}" removed from checklist`);
    }
  
    updateChecklistDisplay() {
      const container = document.getElementById('checklistItems');
      container.innerHTML = '';
      
      this.reportData.checklist.forEach((task, index) => {
        const item = document.createElement('div');
        item.className = 'checklist-item';
        item.innerHTML = `
          <span>✅ ${task}</span>
          <button class="remove-item-btn" onclick="reportGenerator.removeChecklistItem(${index})">❌</button>
        `;
        container.appendChild(item);
      });
    }
  
    // =====================================
    // FORM MANAGEMENT
    // =====================================
    
    updateReportData() {
      this.reportData.location = document.getElementById('location').value;
      this.reportData.severity = parseInt(document.getElementById('severity').value);
      this.reportData.hazards = document.getElementById('hazards').value
        .split(',')
        .map(h => h.trim())
        .filter(h => h.length > 0);
      this.reportData.notes = document.getElementById('notes').value;
      this.reportData.image_url = document.getElementById('imageUrl').value;
      this.reportData.timestamp = new Date().toISOString();
    }
  
    // =====================================
    // VALIDATION
    // =====================================
    
    validateReportData() {
      const errors = [];
      
      if (!this.reportData.location || this.reportData.location.trim().length === 0) {
        errors.push("location");
      }
      
      if (this.reportData.severity < 1 || this.reportData.severity > 10) {
        errors.push("severity level");
      }
      
      if (this.reportData.checklist.length === 0) {
        errors.push("at least one action item");
      }
      
      if (errors.length > 0) {
        let message = "Please provide: " + errors.join(", ");
        return { isValid: false, message: message };
      }
      
      if (this.reportData.hazards.length === 0 && (!this.reportData.notes || this.reportData.notes.trim().length === 0)) {
        return { 
          isValid: false, 
          message: "Please add either hazards or additional notes to create a comprehensive report" 
        };
      }
      
      return { isValid: true, message: "Valid report data" };
    }
  
    // =====================================
    // PDF GENERATION
    // =====================================
    
    async generatePDF() {
      const generateBtn = document.getElementById('generateBtn');
      const statusIndicator = document.getElementById('statusIndicator');
      
      this.updateReportData();
      const validationResult = this.validateReportData();
      
      if (!validationResult.isValid) {
        this.showToast(`⚠️ ${validationResult.message}`, 'warning');
        return;
      }
      
      generateBtn.disabled = true;
      statusIndicator.className = 'status-indicator loading';
      statusIndicator.innerHTML = '<div class="spinner"></div><span>Generating professional PDF report...</span>';
      
      try {
        const response = await fetch('/generate-report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.reportData)
        });
        
        if (response.ok) {
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `emergency_report_${this.reportData.report_id}.pdf`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          window.URL.revokeObjectURL(url);
          
          statusIndicator.className = 'status-indicator success';
          statusIndicator.innerHTML = '✅ <span>PDF generated and downloaded successfully!</span>';
          
          this.showToast('📄 Professional report generated successfully!', 'success');
          this.reportData.report_id = this.generateReportId();
          this.renderPreview();
          
        } else {
          throw new Error(`Server error: ${response.status}`);
        }
      } catch (error) {
        console.error('PDF generation error:', error);
        statusIndicator.className = 'status-indicator error';
        statusIndicator.innerHTML = '❌ <span>PDF generation failed. Please check your connection and try again.</span>';
        this.showToast('❌ PDF generation failed. Please try again.', 'error');
      } finally {
        generateBtn.disabled = false;
        setTimeout(() => {
          statusIndicator.className = 'status-indicator';
        }, 5000);
      }
    }
  
    // =====================================
    // PREVIEW RENDERING
    // =====================================
    
    renderPreview() {
      this.updateReportData();
      
      const preview = document.getElementById('preview');
      const formatCoords = `${this.reportData.coordinates[0].toFixed(6)}, ${this.reportData.coordinates[1].toFixed(6)}`;
      
      const formattedDate = new Date(this.reportData.timestamp).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      });
      
      const severityDisplay = `${this.reportData.severity}/10 ${this.getSeverityEmoji(this.reportData.severity)} ${this.getSeverityLabel(this.reportData.severity)}`;
      
      preview.innerHTML = `
        <div class="preview-field">
          <div class="preview-label">Report ID:</div>
          <div class="preview-value"><code>${this.reportData.report_id}</code></div>
        </div>
        <div class="preview-field">
          <div class="preview-label">Location:</div>
          <div class="preview-value">${this.reportData.location || '<em>Not specified</em>'}</div>
        </div>
        <div class="preview-field">
          <div class="preview-label">Coordinates:</div>
          <div class="preview-value"><code>${formatCoords}</code></div>
        </div>
        <div class="preview-field">
          <div class="preview-label">Severity:</div>
          <div class="preview-value" style="font-weight: 600; color: ${this.getSeverityColor(this.reportData.severity)}">${severityDisplay}</div>
        </div>
        <div class="preview-field">
          <div class="preview-label">Hazards:</div>
          <div class="preview-value">${this.reportData.hazards.length > 0 ? this.reportData.hazards.join(', ') : '<em>None specified</em>'}</div>
        </div>
        <div class="preview-field">
          <div class="preview-label">Notes:</div>
          <div class="preview-value">${this.reportData.notes || '<em>—</em>'}</div>
        </div>
        <div class="preview-field">
          <div class="preview-label">Timestamp:</div>
          <div class="preview-value">${formattedDate}</div>
        </div>
        ${this.renderChecklistPreview()}
        ${this.renderImagePreview()}
      `;
    }
  
    // =====================================
    // TEMPLATES & UTILITIES
    // =====================================
    
    fillTemplate(type) {
      const templates = {
        fire: {
          location: "Main Street, Downtown",
          hazards: ["fire", "smoke", "heat"],
          severity: 8,
          notes: "Active fire observed with heavy smoke. Multiple units responding.",
          checklist: ["Evacuate area", "Contact fire department", "Clear escape routes", "Account for personnel"]
        },
        flood: {
          location: "River Valley Road",
          hazards: ["flooding", "water damage", "debris"],
          severity: 6,
          notes: "Rising water levels due to heavy rainfall. Road access limited.",
          checklist: ["Move to higher ground", "Monitor water levels", "Secure utilities", "Document damage"]
        },
        earthquake: {
          location: "City Center",
          hazards: ["structural damage", "aftershocks", "debris"],
          severity: 9,
          notes: "Major seismic activity. Structural integrity compromised in several buildings.",
          checklist: ["Drop, cover, hold", "Check for injuries", "Inspect building damage", "Locate safe assembly area"]
        },
        accident: {
          location: "Highway 35, Mile Marker 42",
          hazards: ["vehicle damage", "traffic obstruction", "potential injuries"],
          severity: 5,
          notes: "Multi-vehicle collision blocking two lanes. Emergency services notified.",
          checklist: ["Secure scene", "Check for injuries", "Direct traffic", "Document incident"]
        },
        medical: {
          location: "Office Building, 5th Floor",
          hazards: ["medical emergency", "potential contamination"],
          severity: 7,
          notes: "Medical emergency requiring immediate attention. CPR in progress.",
          checklist: ["Call 911", "Perform first aid", "Clear area", "Gather medical history"]
        }
      };
  
      if (templates[type]) {
        const template = templates[type];
        
        document.getElementById('location').value = template.location;
        document.getElementById('hazards').value = template.hazards.join(', ');
        document.getElementById('severity').value = template.severity;
        document.getElementById('severityValue').textContent = template.severity;
        document.getElementById('notes').value = template.notes;
        
        this.reportData.location = template.location;
        this.reportData.hazards = template.hazards;
        this.reportData.severity = template.severity;
        this.reportData.notes = template.notes;
        this.reportData.checklist = [...template.checklist];
        
        this.updateChecklistDisplay();
        this.renderPreview();
        
        this.showToast(`🚀 ${type.charAt(0).toUpperCase() + type.slice(1)} template loaded`);
      }
    }
  
    // =====================================
    // DRAFT MANAGEMENT
    // =====================================
    
    saveDraft() {
      this.updateReportData();
      localStorage.setItem('emergencyReportDraft', JSON.stringify({
        ...this.reportData,
        savedAt: new Date().toISOString()
      }));
      this.showToast('💾 Draft saved successfully', 'success');
    }
  
    loadDraft() {
      const draft = localStorage.getItem('emergencyReportDraft');
      if (!draft) {
        this.showToast('📝 No saved draft found', 'info');
        return;
      }
      
      try {
        const draftData = JSON.parse(draft);
        const timeDiff = new Date() - new Date(draftData.savedAt);
        
        if (timeDiff < 7 * 24 * 60 * 60 * 1000) {
          this.reportData = { ...draftData };
          this.populateFormFromData();
          const savedTime = new Date(draftData.savedAt).toLocaleString();
          this.showToast(`📝 Draft loaded from ${savedTime}`, 'success');
        } else {
          localStorage.removeItem('emergencyReportDraft');
          this.showToast('📝 Old draft found and cleared', 'info');
        }
      } catch (error) {
        console.warn('Failed to load draft:', error);
        this.showToast('❌ Failed to load draft', 'error');
      }
    }
  
    // =====================================
    // HELPER METHODS
    // =====================================
    
    getSeverityEmoji(severity) {
      if (severity <= 3) return '🟢';
      if (severity <= 6) return '🟡';
      if (severity <= 8) return '🟠';
      return '🔴';
    }
  
    getSeverityLabel(severity) {
      if (severity <= 2) return 'Minimal';
      if (severity <= 4) return 'Low';
      if (severity <= 6) return 'Moderate';
      if (severity <= 8) return 'High';
      return 'Critical';
    }
  
    getSeverityColor(severity) {
      if (severity <= 3) return '#10b981';
      if (severity <= 6) return '#f59e0b';
      if (severity <= 8) return '#f97316';
      return '#ef4444';
    }
  
    showToast(message, type = 'info') {
      // Toast implementation
      let toast = document.getElementById('toast');
      if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        toast.style.cssText = `
          position: fixed; bottom: 20px; right: 20px; background: #3b82f6;
          color: white; padding: 1rem 1.5rem; border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); z-index: 1000;
          transform: translateY(100px); opacity: 0; transition: all 0.3s ease;
          max-width: 300px; font-size: 0.9rem;
        `;
        document.body.appendChild(toast);
      }
      
      const colors = {
        info: '#3b82f6', success: '#10b981', warning: '#f59e0b', error: '#ef4444'
      };
      
      toast.style.backgroundColor = colors[type] || colors.info;
      toast.textContent = message;
      toast.style.transform = 'translateY(0)';
      toast.style.opacity = '1';
      
      setTimeout(() => {
        toast.style.transform = 'translateY(100px)';
        toast.style.opacity = '0';
      }, 3000);
    }
  
    // Additional helper methods...
    addEventListeners() {
      document.getElementById('location').addEventListener('input', () => this.renderPreview());
      document.getElementById('severity').addEventListener('input', () => this.renderPreview());
      document.getElementById('hazards').addEventListener('input', () => this.renderPreview());
      document.getElementById('notes').addEventListener('input', () => this.renderPreview());
      document.getElementById('imageUrl').addEventListener('input', () => this.renderPreview());
      
      document.getElementById('newTaskInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          this.addChecklistItem();
        }
      });
    }
  
    addDraftControls() {
      // Implementation for adding draft control buttons
    }
  
    enableAutoSave() {
      setInterval(() => {
        this.updateReportData();
        if (this.reportData.location || this.reportData.hazards.length > 0 || this.reportData.checklist.length > 0) {
          localStorage.setItem('emergencyReportDraft', JSON.stringify({
            ...this.reportData,
            savedAt: new Date().toISOString()
          }));
        }
      }, 30000);
    }
  
    checkForDraft() {
      setTimeout(() => {
        const draft = localStorage.getItem('emergencyReportDraft');
        if (draft) {
          const shouldLoad = confirm('📝 Found a saved draft. Would you like to load it?');
          if (shouldLoad) {
            this.loadDraft();
          }
        }
      }, 1000);
    }
  }
  
  // =====================================
  // GLOBAL INITIALIZATION
  // =====================================
  
  let reportGenerator;
  
  window.addEventListener('load', () => {
    reportGenerator = new EmergencyReportGenerator();
  });
  
  // Global functions for HTML onclick handlers
  window.getCurrentLocation = () => reportGenerator.getCurrentLocation();
  window.searchLocation = () => reportGenerator.searchLocation();
  window.resetMap = () => reportGenerator.resetMap();
  window.addChecklistItem = () => reportGenerator.addChecklistItem();
  window.addQuickTask = (task) => reportGenerator.addQuickTask(task);
  window.removeChecklistItem = (index) => reportGenerator.removeChecklistItem(index);
  window.fillTemplate = (type) => reportGenerator.fillTemplate(type);
  window.clearForm = () => reportGenerator.clearForm();
  window.generatePDF = () => reportGenerator.generatePDF();