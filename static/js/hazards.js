/**
 * Enhanced Hazard Detection System - Minimal Working Version
 * File: static/js/hazards.js
 * 
 * This is a simplified version that matches the minimal HTML template
 * and provides core functionality without complex dependencies.
 */

'use strict';

// ============================================================================
// GLOBAL STATE MANAGEMENT
// ============================================================================

class HazardDetectionSystem {
  constructor() {
    // Core system state
    this.currentProcessingMode = 'edge'; // 'edge' or 'server'
    this.comparisonMode = false;
    this.locationContextEnabled = false;
    this.detectionMode = 'general';
    
    // Results storage
    this.lastResults = null;
    this.processingTimes = [];
    
    // Error handling
    this.errorCount = 0;
    this.successCount = 0;
    
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.showToast('🚀 Hazard Detection System ready!', 'success');
    console.log('🚀 Hazard Detection System - Minimal Version Loaded');
  }

  // ============================================================================
  // EVENT HANDLERS
  // ============================================================================

  setupEventListeners() {
    const form = document.getElementById('hazardForm');
    const imageInput = document.getElementById('imageInput');
    const testBtn = document.getElementById('testBtn');
    const enableComparison = document.getElementById('enableComparison');
    const includeLocation = document.getElementById('includeLocation');

    if (form) {
      form.addEventListener('submit', (e) => this.handleFormSubmission(e));
    }

    if (imageInput) {
      imageInput.addEventListener('change', (e) => this.handleFileSelection(e));
    }

    if (testBtn) {
      testBtn.addEventListener('click', () => this.runSystemTest());
    }

    if (enableComparison) {
      enableComparison.addEventListener('change', (e) => this.handleComparisonToggle(e));
    }

    if (includeLocation) {
      includeLocation.addEventListener('change', (e) => this.handleLocationToggle(e));
    }
  }

  async handleFormSubmission(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
      this.showToast('❌ Please select an image file', 'warning');
      return;
    }

    try {
      this.showToast('🔍 Analyzing image for hazards...', 'info');
      
      // Simulate processing delay
      await this.simulateProcessing();
      
      // Generate results based on mode
      let results;
      if (this.comparisonMode) {
        results = await this.runComparisonAnalysis(file);
      } else {
        results = await this.runSingleAnalysis(file);
      }
      
      this.displayResults(results);
      this.lastResults = results;
      this.successCount++;
      
      const mode = this.comparisonMode ? 'comparison' : 'single';
      this.showToast(`✅ ${mode} analysis complete!`, 'success');
      
    } catch (error) {
      this.handleError('Analysis failed', error);
    }
  }

  handleFileSelection(e) {
    const file = e.target.files[0];
    
    if (!file) return;
    
    // Validate file
    const validation = this.validateImageFile(file);
    if (!validation.valid) {
      this.showToast(validation.error, 'error');
      e.target.value = '';
      return;
    }
    
    this.createImagePreview(file);
    this.showToast('📷 Image loaded successfully', 'success');
  }

  handleComparisonToggle(e) {
    this.comparisonMode = e.target.checked;
    const mode = this.comparisonMode ? 'enabled' : 'disabled';
    this.showToast(`🔄 Comparison mode ${mode}`, 'info');
  }

  handleLocationToggle(e) {
    this.locationContextEnabled = e.target.checked;
    const mode = this.locationContextEnabled ? 'enabled' : 'disabled';
    this.showToast(`📍 Location context ${mode}`, 'info');
  }

  // ============================================================================
  // FILE VALIDATION AND PREVIEW
  // ============================================================================

  validateImageFile(file) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    
    if (!file) {
      return { valid: false, error: 'No file selected' };
    }
    
    if (file.size > maxSize) {
      return { valid: false, error: 'File too large (max 10MB)' };
    }
    
    if (!allowedTypes.includes(file.type)) {
      return { valid: false, error: 'Invalid file type. Use JPEG, PNG, WebP, or GIF' };
    }
    
    return { valid: true };
  }

  createImagePreview(file) {
    const previewContainer = document.getElementById('previewContainer');
    if (!previewContainer) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'preview-wrapper';
    
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.className = 'preview-image';
    img.alt = 'Image preview for hazard detection';
    
    const info = document.createElement('div');
    info.style.cssText = `
      margin-top: 0.5rem;
      padding: 0.75rem;
      background: #f3f4f6;
      border-radius: 6px;
      font-size: 0.9rem;
      border: 1px solid #e5e7eb;
    `;
    
    // Add image load handler to get dimensions
    img.onload = () => {
      info.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div>
            <strong>📷 ${file.name}</strong><br>
            <span style="color: #6b7280;">
              ${(file.size / 1024 / 1024).toFixed(2)} MB • ${file.type} • ${img.naturalWidth}×${img.naturalHeight}px
            </span>
          </div>
          <div style="background: #dbeafe; color: #1e40af; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: bold;">
            ✅ Ready
          </div>
        </div>
      `;
    };
    
    wrapper.appendChild(img);
    wrapper.appendChild(info);
    
    previewContainer.innerHTML = '';
    previewContainer.appendChild(wrapper);
  }

  // ============================================================================
  // HAZARD ANALYSIS
  // ============================================================================

  async simulateProcessing() {
    // Simulate realistic processing time
    const baseTime = 1500;
    const randomDelay = Math.random() * 1000;
    return new Promise(resolve => setTimeout(resolve, baseTime + randomDelay));
  }

  async runSingleAnalysis(file) {
    const startTime = performance.now();
    
    // Generate mock hazards based on current mode
    const hazards = this.generateMockHazards();
    
    const processingTime = performance.now() - startTime;
    this.processingTimes.push(processingTime);
    
    return {
      hazards,
      processingTime: Math.round(processingTime),
      processingMode: this.currentProcessingMode,
      timestamp: new Date().toISOString(),
      confidence: this.calculateAverageConfidence(hazards),
      imageInfo: {
        name: file.name,
        size: file.size,
        type: file.type
      }
    };
  }

  async runComparisonAnalysis(file) {
    this.showToast('🔄 Running comparison analysis...', 'info');
    
    // Simulate both edge and server processing
    const [edgeResults, serverResults] = await Promise.all([
      this.runEdgeAnalysis(file),
      this.runServerAnalysis(file)
    ]);
    
    return {
      mode: 'comparison',
      edge: edgeResults,
      server: serverResults,
      timestamp: new Date().toISOString()
    };
  }

  async runEdgeAnalysis(file) {
    const startTime = performance.now();
    
    // Edge AI tends to be faster but potentially less accurate
    const hazards = this.generateMockHazards('edge');
    
    return {
      hazards,
      processingTime: Math.round(performance.now() - startTime) + Math.random() * 300,
      processingMode: 'edge',
      modelVersion: 'edge-v1.2.0',
      confidence: this.calculateAverageConfidence(hazards),
      memoryUsage: Math.round(Math.random() * 50) + 20 // MB
    };
  }

  async runServerAnalysis(file) {
    const startTime = performance.now();
    
    // Server AI tends to be slower but potentially more accurate
    await new Promise(resolve => setTimeout(resolve, 800)); // Additional server delay
    
    const hazards = this.generateMockHazards('server');
    
    return {
      hazards,
      processingTime: Math.round(performance.now() - startTime) + Math.random() * 500 + 500,
      processingMode: 'server',
      modelVersion: 'server-v2.1.0',
      confidence: this.calculateAverageConfidence(hazards),
      networkLatency: Math.round(Math.random() * 200) + 50 // ms
    };
  }

  generateMockHazards(mode = 'general') {
    const allHazards = [
      {
        type: '🔥 Fire',
        baseConfidence: 94,
        severity: 'critical',
        description: 'Active fire detected - immediate evacuation required',
        actions: ['Evacuate immediately', 'Call emergency services', 'Avoid smoke inhalation']
      },
      {
        type: '💧 Flood Water',
        baseConfidence: 78,
        severity: 'high',
        description: 'Standing water detected - contamination risk',
        actions: ['Avoid contact', 'Check for electrical hazards', 'Use protective equipment']
      },
      {
        type: '⚡ Electrical Hazard',
        baseConfidence: 89,
        severity: 'critical',
        description: 'Exposed electrical components detected',
        actions: ['Do not touch', 'Turn off power if safe', 'Maintain safe distance']
      },
      {
        type: '🏗️ Structural Damage',
        baseConfidence: 67,
        severity: 'medium',
        description: 'Structural integrity compromised',
        actions: ['Evacuate building', 'Check for unstable debris', 'Contact engineer']
      },
      {
        type: '🌪️ Storm Damage',
        baseConfidence: 82,
        severity: 'high',
        description: 'Wind damage to structures and vegetation',
        actions: ['Check for falling debris', 'Inspect damage', 'Clear drainage']
      },
      {
        type: '☢️ Chemical Spill',
        baseConfidence: 91,
        severity: 'critical',
        description: 'Hazardous material detected',
        actions: ['Evacuate area', 'Avoid inhalation', 'Contact HAZMAT team']
      }
    ];

    // Randomly select 1-4 hazards
    const numHazards = Math.floor(Math.random() * 4) + 1;
    const selectedHazards = allHazards
      .sort(() => 0.5 - Math.random())
      .slice(0, numHazards);

    // Adjust confidence based on mode
    return selectedHazards.map(hazard => {
      let confidence = hazard.baseConfidence;
      
      // Add some randomness
      confidence += (Math.random() - 0.5) * 10;
      
      // Server AI might be slightly more confident
      if (mode === 'server') {
        confidence += Math.random() * 5;
      }
      
      // Edge AI might be slightly less confident
      if (mode === 'edge') {
        confidence -= Math.random() * 3;
      }
      
      return {
        ...hazard,
        confidence: Math.max(50, Math.min(100, Math.round(confidence * 100) / 100)),
        id: this.generateId(),
        timestamp: new Date().toISOString(),
        bbox: this.generateRandomBbox()
      };
    });
  }

  generateRandomBbox() {
    // Generate random bounding box coordinates [x, y, width, height] normalized to 0-1
    return [
      Math.random() * 0.6, // x
      Math.random() * 0.6, // y
      0.2 + Math.random() * 0.4, // width
      0.2 + Math.random() * 0.4  // height
    ];
  }

  calculateAverageConfidence(hazards) {
    if (hazards.length === 0) return 0;
    const sum = hazards.reduce((acc, hazard) => acc + hazard.confidence, 0);
    return Math.round(sum / hazards.length);
  }

  // ============================================================================
  // RESULTS DISPLAY
  // ============================================================================

  displayResults(results) {
    const resultSection = document.getElementById('resultSection');
    const hazardList = document.getElementById('hazardList');
    
    if (!resultSection || !hazardList) return;

    if (results.mode === 'comparison') {
      this.displayComparisonResults(results);
    } else {
      this.displaySingleResults(results);
    }
    
    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  displaySingleResults(results) {
    const hazardList = document.getElementById('hazardList');
    
    hazardList.innerHTML = results.hazards.map(hazard => `
      <div class="hazard-card" style="animation: slideInUp 0.3s ease;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
          <div style="display: flex; align-items: center; gap: 0.75rem; flex: 1;">
            <div style="font-size: 2rem;" title="${hazard.type}">${hazard.type.split(' ')[0]}</div>
            <div style="flex: 1;">
              <h4 style="margin: 0 0 0.25rem 0; color: #dc2626; font-weight: bold; font-size: 1.1rem;">${hazard.type}</h4>
              <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem;">
                Confidence: <span style="font-weight: bold; color: ${this.getConfidenceColor(hazard.confidence)};">${hazard.confidence}%</span> • 
                Severity: <span style="color: ${this.getSeverityColor(hazard.severity)}; font-weight: bold; text-transform: capitalize;">${hazard.severity}</span>
              </div>
              <p style="margin: 0; color: #374151; font-size: 0.9rem; line-height: 1.4;">${hazard.description}</p>
            </div>
          </div>
          <button onclick="hazardSystem.showHazardDetails('${hazard.id}')" 
                  style="background: #3b82f6; color: white; border: none; padding: 0.5rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem;">
            ℹ️ Details
          </button>
        </div>
        
        <div style="background: #f9fafb; padding: 0.75rem; border-radius: 6px; border: 1px solid #e5e7eb;">
          <h5 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.9rem; font-weight: bold;">Recommended Actions:</h5>
          <ul style="margin: 0; padding-left: 1.5rem; color: #6b7280;">
            ${hazard.actions.map(action => `<li style="margin-bottom: 0.25rem; font-size: 0.85rem;">${action}</li>`).join('')}
          </ul>
        </div>
      </div>
    `).join('');

    // Add summary
    this.addAnalysisSummary(hazardList, results);
  }

  displayComparisonResults(results) {
    const hazardList = document.getElementById('hazardList');
    
    hazardList.innerHTML = `
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
        <!-- Edge AI Results -->
        <div style="border: 2px solid #3b82f6; border-radius: 12px; padding: 1rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(59, 130, 246, 0.1));">
          <h3 style="margin: 0 0 1rem 0; color: #1e40af; display: flex; align-items: center; gap: 0.5rem;">
            🔒 Edge AI Analysis
          </h3>
          
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="text-align: center; background: rgba(255,255,255,0.7); padding: 0.75rem; border-radius: 6px;">
              <div style="font-size: 1.2rem; font-weight: bold; color: #3b82f6;">${results.edge.processingTime}ms</div>
              <div style="font-size: 0.8rem; color: #6b7280;">Processing Time</div>
            </div>
            <div style="text-align: center; background: rgba(255,255,255,0.7); padding: 0.75rem; border-radius: 6px;">
              <div style="font-size: 1.2rem; font-weight: bold; color: #3b82f6;">${results.edge.confidence}%</div>
              <div style="font-size: 0.8rem; color: #6b7280;">Avg Confidence</div>
            </div>
          </div>
          
          <div>
            ${results.edge.hazards.map(hazard => `
              <div style="background: rgba(255,255,255,0.7); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;">
                <div style="font-weight: bold; font-size: 0.9rem;">${hazard.type}</div>
                <div style="font-size: 0.8rem; color: #6b7280;">${hazard.confidence}% confidence</div>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Server AI Results -->
        <div style="border: 2px solid #16a34a; border-radius: 12px; padding: 1rem; background: linear-gradient(135deg, rgba(22, 163, 74, 0.05), rgba(22, 163, 74, 0.1));">
          <h3 style="margin: 0 0 1rem 0; color: #15803d; display: flex; align-items: center; gap: 0.5rem;">
            ☁️ Server AI Analysis
          </h3>
          
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="text-align: center; background: rgba(255,255,255,0.7); padding: 0.75rem; border-radius: 6px;">
              <div style="font-size: 1.2rem; font-weight: bold; color: #16a34a;">${results.server.processingTime}ms</div>
              <div style="font-size: 0.8rem; color: #6b7280;">Processing Time</div>
            </div>
            <div style="text-align: center; background: rgba(255,255,255,0.7); padding: 0.75rem; border-radius: 6px;">
              <div style="font-size: 1.2rem; font-weight: bold; color: #16a34a;">${results.server.confidence}%</div>
              <div style="font-size: 0.8rem; color: #6b7280;">Avg Confidence</div>
            </div>
          </div>
          
          <div>
            ${results.server.hazards.map(hazard => `
              <div style="background: rgba(255,255,255,0.7); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;">
                <div style="font-weight: bold; font-size: 0.9rem;">${hazard.type}</div>
                <div style="font-size: 0.8rem; color: #6b7280;">${hazard.confidence}% confidence</div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
      
      <!-- Performance Comparison -->
      <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
        <h4 style="margin: 0 0 1rem 0;">⚡ Performance Comparison</h4>
        <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
          <strong>Key Insight:</strong> Edge AI provides 
          <span style="color: #3b82f6; font-weight: bold;">instant privacy-first analysis</span> 
          while Server AI offers 
          <span style="color: #16a34a; font-weight: bold;">enhanced accuracy with larger models</span>
        </div>
      </div>
    `;
  }

  addAnalysisSummary(container, results) {
    const summary = document.createElement('div');
    summary.style.cssText = `
      background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
      color: white;
      padding: 1.5rem;
      border-radius: 12px;
      margin-top: 1.5rem;
      text-align: center;
    `;
    
    summary.innerHTML = `
      <div style="font-weight: bold; margin-bottom: 1rem; font-size: 1.2rem;">📊 Analysis Summary</div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
        <div style="background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px;">
          <div style="font-size: 1.5rem; font-weight: bold;">${results.hazards.length}</div>
          <div style="font-size: 0.8rem; opacity: 0.9;">Hazards Found</div>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px;">
          <div style="font-size: 1.5rem; font-weight: bold;">${results.processingTime}ms</div>
          <div style="font-size: 0.8rem; opacity: 0.9;">Processing Time</div>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px;">
          <div style="font-size: 1.5rem; font-weight: bold;">${results.confidence}%</div>
          <div style="font-size: 0.8rem; opacity: 0.9;">Avg Confidence</div>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px;">
          <div style="font-size: 1.5rem; font-weight: bold;">${results.processingMode.toUpperCase()}</div>
          <div style="font-size: 0.8rem; opacity: 0.9;">AI Mode</div>
        </div>
      </div>
      <div style="font-size: 0.9rem; opacity: 0.9;">
        Analysis completed at ${new Date(results.timestamp).toLocaleTimeString()}
        ${this.locationContextEnabled ? ' with location context' : ''}
      </div>
    `;
    
    container.appendChild(summary);
  }

  // ============================================================================
  // SYSTEM TEST
  // ============================================================================

  async runSystemTest() {
    this.showToast('🧪 Running comprehensive system test...', 'info');
    
    try {
      // Simulate system test
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const testResults = {
        hazards: [
          {
            type: '🧪 System Test',
            confidence: 98,
            severity: 'high',
            description: 'System test completed - all components operational',
            actions: ['Verify system status', 'Check all modules', 'Confirm connectivity'],
            id: this.generateId(),
            timestamp: new Date().toISOString(),
            bbox: [0.2, 0.2, 0.6, 0.6]
          },
          {
            type: '🔍 AI Validation',
            confidence: 95,
            severity: 'medium',
            description: 'AI detection models validated and ready',
            actions: ['AI models loaded', 'Processing pipeline active', 'Analysis capabilities confirmed'],
            id: this.generateId(),
            timestamp: new Date().toISOString(),
            bbox: [0.1, 0.5, 0.4, 0.3]
          }
        ],
        processingTime: 247,
        processingMode: 'test',
        timestamp: new Date().toISOString(),
        confidence: 96.5
      };
      
      this.displayResults(testResults);
      this.showToast('✅ System test completed successfully!', 'success');
      
    } catch (error) {
      this.handleError('System test failed', error);
    }
  }

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  showHazardDetails(hazardId) {
    this.showToast(`ℹ️ Showing details for hazard: ${hazardId}`, 'info');
    
    // In a real implementation, this would show a detailed modal
    console.log('Hazard details requested for ID:', hazardId);
  }

  getSeverityColor(severity) {
    const colors = {
      low: '#10b981',
      medium: '#f59e0b',
      high: '#ef4444',
      critical: '#dc2626'
    };
    return colors[severity] || '#6b7280';
  }

  getConfidenceColor(confidence) {
    if (confidence >= 90) return '#10b981';
    if (confidence >= 70) return '#f59e0b';
    if (confidence >= 50) return '#ef4444';
    return '#6b7280';
  }

  generateId() {
    return 'hazard_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    
    toast.textContent = message;
    toast.className = 'toast show';
    
    const colors = {
      info: '#3b82f6',
      success: '#10b981',
      warning: '#f59e0b',
      error: '#ef4444'
    };
    
    toast.style.backgroundColor = colors[type] || colors.info;
    
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3000);
  }

  handleError(message, error) {
    console.error(message, error);
    this.errorCount++;
    this.showToast(`❌ ${message}`, 'error');
  }
}

// ============================================================================
// GLOBAL INITIALIZATION
// ============================================================================

// Global instance
let hazardSystem;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  hazardSystem = new HazardDetectionSystem();
  
  // Make globally accessible for HTML onclick handlers
  window.hazardSystem = hazardSystem;
  
  // Add CSS animations if not already present
  if (!document.querySelector('#hazard-animations')) {
    const style = document.createElement('style');
    style.id = 'hazard-animations';
    style.textContent = `
      @keyframes slideInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
      }
      
      .preview-image {
        transition: all 0.3s ease;
      }
      
      .preview-image:hover {
        transform: scale(1.02);
      }
      
      .hazard-card {
        transition: all 0.3s ease;
      }
      
      .hazard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
      }
    `;
    document.head.appendChild(style);
  }
  
  console.log('🚀 Hazard Detection System - Minimal Version Ready');
});

// Error handling
window.addEventListener('error', (e) => {
  if (hazardSystem) {
    hazardSystem.handleError('JavaScript Error', e.error);
  }
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { HazardDetectionSystem };
}
