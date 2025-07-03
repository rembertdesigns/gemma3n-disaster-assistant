// Enhanced Hazard Detection System
class HazardDetectionSystem {
  constructor() {
    this.imageElement = null;
    this.lastDetections = [];
    this.isProcessing = false;
    this.settings = {
      theme: 'light',
      showBoxes: true,
      soundAlerts: true,
      confidenceThreshold: 0.5,
      maxFileSize: 10 * 1024 * 1024, // 10MB
      autoDetect: false,
      saveHistory: true
    };
    this.detectionHistory = [];
    this.audioContext = null;
    this.supportedFormats = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    
    this.initializeElements();
    this.setupEventListeners();
    this.loadSettings();
    this.initializeAudio();
    this.setupDragAndDrop();
  }

  initializeElements() {
    this.elements = {
      form: document.getElementById("hazardForm"),
      input: document.getElementById("imageInput"),
      previewContainer: document.getElementById("previewContainer"),
      resultSection: document.getElementById("resultSection"),
      canvas: document.getElementById("resultCanvas"),
      hazardList: document.getElementById("hazardList"),
      toast: document.getElementById("toast"),
      testBtn: document.getElementById("runTestHazardBtn"),
      progressBar: document.getElementById("progressBar"),
      toggleBoxes: document.getElementById("toggleBoxes"),
      downloadBtn: document.getElementById("downloadCanvasBtn"),
      openDrawerBtn: document.getElementById("openDrawerBtn"),
      closeDrawerBtn: document.getElementById("closeDrawerBtn"),
      settingsDrawer: document.getElementById("settingsDrawer"),
      drawerToggleTheme: document.getElementById("drawerToggleTheme"),
      drawerToggleBoxes: document.getElementById("drawerToggleBoxes"),
      drawerSoundAlerts: document.getElementById("drawerSoundAlerts"),
      confidenceSlider: document.getElementById("confidenceSlider"),
      autoDetectToggle: document.getElementById("autoDetectToggle"),
      historyBtn: document.getElementById("historyBtn"),
      clearHistoryBtn: document.getElementById("clearHistoryBtn"),
      exportBtn: document.getElementById("exportBtn"),
      fullscreenBtn: document.getElementById("fullscreenBtn"),
      retakeBtn: document.getElementById("retakeBtn"),
      shareBtn: document.getElementById("shareBtn")
    };
  }

  setupEventListeners() {
    // Form submission
    this.elements.form?.addEventListener("submit", this.handleFormSubmit.bind(this));
    
    // File input change
    this.elements.input?.addEventListener("change", this.handleFileChange.bind(this));
    
    // Test button
    this.elements.testBtn?.addEventListener("click", this.runMockDetection.bind(this));
    
    // Download button
    this.elements.downloadBtn?.addEventListener("click", this.downloadResults.bind(this));
    
    // Settings drawer
    this.elements.openDrawerBtn?.addEventListener("click", this.openSettings.bind(this));
    this.elements.closeDrawerBtn?.addEventListener("click", this.closeSettings.bind(this));
    
    // Settings toggles
    this.elements.drawerToggleTheme?.addEventListener("change", this.toggleTheme.bind(this));
    this.elements.drawerToggleBoxes?.addEventListener("change", this.toggleBoxes.bind(this));
    this.elements.drawerSoundAlerts?.addEventListener("change", this.toggleSoundAlerts.bind(this));
    this.elements.toggleBoxes?.addEventListener("change", this.redrawResults.bind(this));
    
    // Advanced controls
    this.elements.confidenceSlider?.addEventListener("input", this.updateConfidenceThreshold.bind(this));
    this.elements.autoDetectToggle?.addEventListener("change", this.toggleAutoDetect.bind(this));
    this.elements.historyBtn?.addEventListener("click", this.showHistory.bind(this));
    this.elements.clearHistoryBtn?.addEventListener("click", this.clearHistory.bind(this));
    this.elements.exportBtn?.addEventListener("click", this.exportResults.bind(this));
    this.elements.fullscreenBtn?.addEventListener("click", this.toggleFullscreen.bind(this));
    this.elements.retakeBtn?.addEventListener("click", this.retakeImage.bind(this));
    this.elements.shareBtn?.addEventListener("click", this.shareResults.bind(this));
    
    // Keyboard shortcuts
    document.addEventListener("keydown", this.handleKeyboardShortcuts.bind(this));
    
    // Window events
    window.addEventListener("beforeunload", this.saveSettings.bind(this));
    window.addEventListener("resize", this.handleWindowResize.bind(this));
  }

  setupDragAndDrop() {
    const dropZone = this.elements.previewContainer;
    if (!dropZone) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, this.preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    dropZone.addEventListener('drop', this.handleDrop.bind(this), false);
  }

  preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
      const file = files[0];
      if (this.validateFile(file)) {
        this.processFile(file);
      }
    }
  }

  async handleFormSubmit(e) {
    e.preventDefault();
    
    if (this.isProcessing) {
      this.showToast("⏳ Detection already in progress...", "warning");
      return;
    }

    if (!this.elements.input?.files[0]) {
      this.showToast("📁 Please select an image to detect hazards.", "error");
      return;
    }

    await this.detectHazards(this.elements.input.files[0]);
  }

  handleFileChange() {
    const file = this.elements.input?.files[0];
    if (file && this.validateFile(file)) {
      this.processFile(file);
    }
  }

  validateFile(file) {
    if (!this.supportedFormats.includes(file.type)) {
      this.showToast("❌ Unsupported file format. Please use JPEG, PNG, WebP, or GIF.", "error");
      return false;
    }

    if (file.size > this.settings.maxFileSize) {
      this.showToast(`📏 File too large. Maximum size is ${this.settings.maxFileSize / (1024 * 1024)}MB.`, "error");
      return false;
    }

    return true;
  }

  processFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      this.loadImage(e.target.result);
    };
    reader.onerror = () => {
      this.showToast("❌ Error reading file.", "error");
    };
    reader.readAsDataURL(file);
  }

  loadImage(src) {
    this.elements.previewContainer.innerHTML = "";
    this.imageElement = new Image();
    this.imageElement.src = src;
    this.imageElement.onload = () => {
      this.elements.previewContainer.appendChild(this.imageElement);
      this.addImageControls();
      
      if (this.settings.autoDetect) {
        setTimeout(() => this.detectHazards(), 1000);
      }
    };
    this.imageElement.onerror = () => {
      this.showToast("❌ Error loading image.", "error");
    };
  }

  addImageControls() {
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'image-controls';
    controlsDiv.innerHTML = `
      <button type="button" class="btn btn-sm" onclick="hazardSystem.rotateImage()">↻ Rotate</button>
      <button type="button" class="btn btn-sm" onclick="hazardSystem.flipImage()">⟷ Flip</button>
      <button type="button" class="btn btn-sm" onclick="hazardSystem.resetImage()">↺ Reset</button>
    `;
    this.elements.previewContainer.appendChild(controlsDiv);
  }

  async detectHazards(file = null) {
    if (!file && !this.elements.input?.files[0]) {
      this.showToast("📁 No image selected.", "error");
      return;
    }

    const fileToProcess = file || this.elements.input.files[0];
    this.isProcessing = true;
    this.setProgress(10);

    const formData = new FormData();
    formData.append("file", fileToProcess);
    formData.append("confidence_threshold", this.settings.confidenceThreshold);

    this.showToast("🔍 Analyzing image for hazards...", "info");

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch("/detect-hazards", {
        method: "POST",
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      this.setProgress(70);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        this.showToast(data.error, "error");
        this.setProgress(0);
        return;
      }

      this.lastDetections = data.predictions || [];
      this.processDetectionResults(data);
      this.setProgress(100);

      if (this.settings.saveHistory) {
        this.saveToHistory(data, fileToProcess.name);
      }

    } catch (err) {
      console.error("Detection error:", err);
      if (err.name === 'AbortError') {
        this.showToast("⏱️ Detection timeout. Please try again.", "error");
      } else {
        this.showToast("❌ Detection failed. Please try again.", "error");
      }
      this.setProgress(0);
    } finally {
      this.isProcessing = false;
    }
  }

  processDetectionResults(data) {
    if (!this.imageElement) {
      this.showToast("❌ No image loaded.", "error");
      return;
    }

    this.drawResults(data);
    this.updateStatistics(data);
    
    if (this.settings.soundAlerts && data.predictions?.length > 0) {
      this.playAlertSound(data.predictions.length);
    }

    // Show detailed analysis
    this.showDetailedAnalysis(data);
  }

  drawResults(data) {
    this.elements.resultSection.style.display = "block";
    const canvas = this.elements.canvas;
    canvas.width = this.imageElement.width;
    canvas.height = this.imageElement.height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(this.imageElement, 0, 0, canvas.width, canvas.height);

    this.elements.hazardList.innerHTML = "";

    if (Array.isArray(data.predictions) && data.predictions.length > 0) {
      data.predictions.forEach((prediction, index) => {
        const { label, score, box } = prediction;
        const [x1, y1, x2, y2] = box;

        if (this.settings.showBoxes) {
          this.drawBoundingBox(ctx, x1, y1, x2, y2, label, score, index);
        }

        this.addHazardToList(label, score, index);
      });

      this.showToast(`🚨 ${data.predictions.length} hazard(s) detected!`, "warning");
    } else {
      this.showToast("✅ No hazards detected.", "success");
    }
  }

  drawBoundingBox(ctx, x1, y1, x2, y2, label, score, index) {
    const colors = ["#dc2626", "#ea580c", "#d97706", "#65a30d", "#059669", "#0891b2"];
    const color = colors[index % colors.length];

    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw label background
    const labelText = `${label} (${(score * 100).toFixed(1)}%)`;
    ctx.font = "bold 14px Arial";
    const textWidth = ctx.measureText(labelText).width;
    
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 25, textWidth + 10, 20);

    // Draw label text
    ctx.fillStyle = "white";
    ctx.fillText(labelText, x1 + 5, y1 - 8);

    // Draw confidence indicator
    const confidenceWidth = (x2 - x1) * score;
    ctx.fillStyle = `${color}80`;
    ctx.fillRect(x1, y2 - 5, confidenceWidth, 3);
  }

  addHazardToList(label, score, index) {
    const li = document.createElement("li");
    li.className = "hazard-item";
    li.innerHTML = `
      <div class="hazard-info">
        <span class="hazard-icon">🚨</span>
        <div class="hazard-details">
          <strong>${label}</strong>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${score * 100}%"></div>
            <span class="confidence-text">${(score * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
      <div class="hazard-actions">
        <button onclick="hazardSystem.focusHazard(${index})" class="btn btn-sm">Focus</button>
        <button onclick="hazardSystem.reportFalsePositive(${index})" class="btn btn-sm">Report</button>
      </div>
    `;
    this.elements.hazardList.appendChild(li);
  }

  updateStatistics(data) {
    const stats = {
      totalDetections: data.predictions?.length || 0,
      highConfidence: data.predictions?.filter(p => p.score > 0.8).length || 0,
      averageConfidence: data.predictions?.length > 0 
        ? (data.predictions.reduce((sum, p) => sum + p.score, 0) / data.predictions.length * 100).toFixed(1)
        : 0,
      processingTime: data.processing_time || 0
    };

    this.displayStatistics(stats);
  }

  displayStatistics(stats) {
    const statsDiv = document.createElement('div');
    statsDiv.className = 'detection-stats';
    statsDiv.innerHTML = `
      <h4>Detection Statistics</h4>
      <div class="stats-grid">
        <div class="stat-item">
          <span class="stat-label">Total Detections:</span>
          <span class="stat-value">${stats.totalDetections}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">High Confidence:</span>
          <span class="stat-value">${stats.highConfidence}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Average Confidence:</span>
          <span class="stat-value">${stats.averageConfidence}%</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Processing Time:</span>
          <span class="stat-value">${stats.processingTime}ms</span>
        </div>
      </div>
    `;
    
    // Insert after hazard list
    this.elements.hazardList.parentNode.insertBefore(statsDiv, this.elements.hazardList.nextSibling);
  }

  showDetailedAnalysis(data) {
    if (!data.predictions || data.predictions.length === 0) return;

    const analysisDiv = document.createElement('div');
    analysisDiv.className = 'detailed-analysis';
    analysisDiv.innerHTML = `
      <h4>Risk Assessment</h4>
      <div class="risk-summary">
        ${this.generateRiskSummary(data.predictions)}
      </div>
      <div class="recommendations">
        <h5>Recommended Actions:</h5>
        <ul>
          ${this.generateRecommendations(data.predictions).map(rec => `<li>${rec}</li>`).join('')}
        </ul>
      </div>
    `;
    
    this.elements.resultSection.appendChild(analysisDiv);
  }

  generateRiskSummary(predictions) {
    const riskLevels = {
      'Downed Power Line': 'CRITICAL',
      'Flood': 'HIGH',
      'Fire': 'CRITICAL',
      'Debris': 'MEDIUM',
      'Structural Damage': 'HIGH'
    };

    const highestRisk = predictions.reduce((max, pred) => {
      const risk = riskLevels[pred.label] || 'LOW';
      return this.getRiskLevel(risk) > this.getRiskLevel(max) ? risk : max;
    }, 'LOW');

    const colors = {
      'CRITICAL': '#dc2626',
      'HIGH': '#ea580c',
      'MEDIUM': '#d97706',
      'LOW': '#65a30d'
    };

    return `
      <div class="risk-indicator" style="background-color: ${colors[highestRisk]}20; border-left: 4px solid ${colors[highestRisk]};">
        <strong>Overall Risk Level: ${highestRisk}</strong>
        <p>Based on ${predictions.length} detected hazard(s)</p>
      </div>
    `;
  }

  generateRecommendations(predictions) {
    const recommendations = [];
    
    predictions.forEach(pred => {
      switch (pred.label) {
        case 'Downed Power Line':
          recommendations.push('🚨 Stay at least 30 feet away from downed power lines');
          recommendations.push('📞 Contact emergency services immediately');
          break;
        case 'Flood':
          recommendations.push('🌊 Avoid walking or driving through flood water');
          recommendations.push('🏠 Move to higher ground if possible');
          break;
        case 'Fire':
          recommendations.push('🔥 Evacuate the area immediately');
          recommendations.push('📱 Call fire department');
          break;
        case 'Debris':
          recommendations.push('🚧 Use caution when navigating around debris');
          recommendations.push('🦺 Wear protective equipment if cleanup is necessary');
          break;
        default:
          recommendations.push('⚠️ Exercise caution in the affected area');
      }
    });

    return [...new Set(recommendations)]; // Remove duplicates
  }

  getRiskLevel(risk) {
    const levels = { 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4 };
    return levels[risk] || 1;
  }

  runMockDetection() {
    this.setProgress(20);
    this.showToast("🧠 Running simulation...", "info");
    
    const mockDetections = [
      { label: "Downed Power Line", score: 0.91, box: [50, 40, 300, 160] },
      { label: "Flood", score: 0.87, box: [100, 200, 280, 320] },
      { label: "Debris", score: 0.73, box: [200, 100, 350, 180] }
    ];

    const mockImage = new Image();
    mockImage.src = "/static/mock_hazard_image.jpg";
    mockImage.onload = () => {
      this.imageElement = mockImage;
      this.elements.previewContainer.innerHTML = "";
      this.elements.previewContainer.appendChild(mockImage);
      
      this.setProgress(80);
      this.processDetectionResults({ 
        predictions: mockDetections,
        processing_time: 1250
      });
      this.setProgress(100);
      this.showToast("✅ Mock detection complete", "success");
    };
    
    mockImage.onerror = () => {
      this.showToast("❌ Failed to load mock image", "error");
      this.setProgress(0);
    };
  }

  // Settings and utility functions
  loadSettings() {
    const saved = localStorage.getItem('hazardDetectionSettings');
    if (saved) {
      this.settings = { ...this.settings, ...JSON.parse(saved) };
    }
    this.applySettings();
  }

  saveSettings() {
    localStorage.setItem('hazardDetectionSettings', JSON.stringify(this.settings));
  }

  applySettings() {
    document.documentElement.setAttribute("data-theme", this.settings.theme);
    if (this.elements.drawerToggleTheme) {
      this.elements.drawerToggleTheme.checked = this.settings.theme === 'dark';
    }
    if (this.elements.drawerToggleBoxes) {
      this.elements.drawerToggleBoxes.checked = this.settings.showBoxes;
    }
    if (this.elements.toggleBoxes) {
      this.elements.toggleBoxes.checked = this.settings.showBoxes;
    }
    if (this.elements.drawerSoundAlerts) {
      this.elements.drawerSoundAlerts.checked = this.settings.soundAlerts;
    }
    if (this.elements.confidenceSlider) {
      this.elements.confidenceSlider.value = this.settings.confidenceThreshold;
    }
  }

  initializeAudio() {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
      console.warn('Audio context not supported');
    }
  }

  playAlertSound(count) {
    if (!this.audioContext || !this.settings.soundAlerts) return;

    const oscillator = this.audioContext.createOscillator();
    const gainNode = this.audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(this.audioContext.destination);
    
    oscillator.frequency.setValueAtTime(800, this.audioContext.currentTime);
    gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
    
    oscillator.start();
    oscillator.stop(this.audioContext.currentTime + 0.2);
    
    // Play multiple beeps for multiple hazards
    for (let i = 1; i < Math.min(count, 3); i++) {
      setTimeout(() => {
        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();
        osc.connect(gain);
        gain.connect(this.audioContext.destination);
        osc.frequency.setValueAtTime(1000, this.audioContext.currentTime);
        gain.gain.setValueAtTime(0.1, this.audioContext.currentTime);
        osc.start();
        osc.stop(this.audioContext.currentTime + 0.2);
      }, i * 300);
    }
  }

  showToast(message, type = "info", duration = 3000) {
    const toast = this.elements.toast;
    if (!toast) return;

    toast.textContent = message;
    toast.className = `toast toast-${type}`;
    toast.style.display = "block";
    toast.style.opacity = "1";
    
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => {
        toast.style.display = "none";
      }, 300);
    }, duration);
  }

  setProgress(percent) {
    const progressBar = this.elements.progressBar;
    if (!progressBar) return;

    progressBar.style.width = `${percent}%`;
    progressBar.parentElement.style.display = percent < 100 ? "block" : "none";
    
    if (percent === 100) {
      setTimeout(() => {
        progressBar.parentElement.style.display = "none";
      }, 1000);
    }
  }

  // Event handlers
  toggleTheme() {
    this.settings.theme = this.elements.drawerToggleTheme.checked ? 'dark' : 'light';
    document.documentElement.setAttribute("data-theme", this.settings.theme);
    this.saveSettings();
  }

  toggleBoxes() {
    this.settings.showBoxes = this.elements.drawerToggleBoxes.checked;
    if (this.elements.toggleBoxes) {
      this.elements.toggleBoxes.checked = this.settings.showBoxes;
    }
    this.redrawResults();
    this.saveSettings();
  }

  toggleSoundAlerts() {
    this.settings.soundAlerts = this.elements.drawerSoundAlerts.checked;
    this.saveSettings();
  }

  redrawResults() {
    if (this.lastDetections.length > 0) {
      this.drawResults({ predictions: this.lastDetections });
    }
  }

  updateConfidenceThreshold() {
    this.settings.confidenceThreshold = parseFloat(this.elements.confidenceSlider.value);
    this.saveSettings();
  }

  openSettings() {
    this.elements.settingsDrawer?.classList.add("open");
  }

  closeSettings() {
    this.elements.settingsDrawer?.classList.remove("open");
  }

  downloadResults() {
    const canvas = this.elements.canvas;
    if (!canvas) return;

    const link = document.createElement("a");
    link.download = `hazard_detection_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  handleKeyboardShortcuts(e) {
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case 'o':
          e.preventDefault();
          this.elements.input?.click();
          break;
        case 's':
          e.preventDefault();
          this.downloadResults();
          break;
        case 'd':
          e.preventDefault();
          this.runMockDetection();
          break;
      }
    }
  }

  // Additional utility methods
  saveToHistory(data, filename) {
    const historyItem = {
      timestamp: new Date().toISOString(),
      filename,
      detections: data.predictions?.length || 0,
      data: data
    };
    
    this.detectionHistory.unshift(historyItem);
    if (this.detectionHistory.length > 50) {
      this.detectionHistory = this.detectionHistory.slice(0, 50);
    }
    
    localStorage.setItem('hazardDetectionHistory', JSON.stringify(this.detectionHistory));
  }

  showHistory() {
    const history = JSON.parse(localStorage.getItem('hazardDetectionHistory') || '[]');
    console.log('Detection History:', history);
    // Implementation for showing history modal would go here
  }

  clearHistory() {
    localStorage.removeItem('hazardDetectionHistory');
    this.detectionHistory = [];
    this.showToast("🗑️ History cleared", "info");
  }

  focusHazard(index) {
    const detection = this.lastDetections[index];
    if (detection) {
      const [x1, y1, x2, y2] = detection.box;
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      
      // Highlight the specific detection
      this.highlightDetection(index);
      this.showToast(`🎯 Focused on ${detection.label}`, "info");
    }
  }

  highlightDetection(index) {
    const canvas = this.elements.canvas;
    const ctx = canvas.getContext('2d');
    
    // Redraw with highlighted detection
    this.drawResults({ predictions: this.lastDetections });
    
    // Add highlight effect
    const detection = this.lastDetections[index];
    const [x1, y1, x2, y2] = detection.box;
    
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 5;
    ctx.strokeRect(x1 - 2, y1 - 2, x2 - x1 + 4, y2 - y1 + 4);
  }

  reportFalsePositive(index) {
    const detection = this.lastDetections[index];
    if (detection) {
      // In a real app, this would send feedback to the server
      console.log('False positive reported:', detection);
      this.showToast("📝 False positive reported. Thank you for your feedback!", "success");
    }
  }
}

// Initialize the system when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.hazardSystem = new HazardDetectionSystem();
});

// Service Worker Registration
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/static/js/sw.js")
    .then(() => console.log("✅ Service Worker registered"))
    .catch((err) => console.error("❌ SW registration failed:", err));
}

// Export for external use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = HazardDetectionSystem;
}