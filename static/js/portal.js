/**
 * portal.js - Professional Enhanced Emergency Portal
 * Optimized for real emergency situations with seamless UX
 */

(function() {
    "use strict";

    // --- ENHANCED GLOBAL STATE ---
    const portal = {
        // Core state
        currentStep: 0,
        userLocation: null,
        isVoiceRecording: false,
        mediaRecorder: null,
        audioChunks: [],
        recognition: null,
        
        // Enhanced permissions tracking
        permissionsGranted: { 
            location: false, 
            microphone: false, 
            camera: false, 
            notifications: false 
        },
        
        // Smart form state
        formData: {
            reportMethod: 'text',
            priority: 'medium',
            description: '',
            location: '',
            evidence: null,
            autoSave: true
        },
        
        // UX enhancement state
        isFormValid: false,
        hasUnsavedChanges: false,
        currentAnalysis: {},
        userReports: [],
        emergencyContacts: [],
        
        // Performance tracking
        startTime: Date.now(),
        stepTimes: {},
        userInteractions: []
    };

    // --- ENHANCED INITIALIZATION ---
    function initializeApp() {
        console.log("🚀 Professional Portal Initializing...");
        try {
            // Core setup
            setupEventListeners();
            setupKeyboardShortcuts();
            setupSmartDefaults();
            setupAutoSave();
            setupProgressiveEnhancement();
            
            // Permission and device checks
            portal.checkExistingPermissions();
            portal.detectUserCapabilities();
            
            // AI and voice setup
            portal.setupAdvancedSpeechRecognition();
            portal.updateAIStatus('ready');
            
            // UX enhancements
            portal.setupSmartNavigation();
            portal.setupFormEnhancements();
            portal.loadUserPreferences();
            
            // Initial state
            portal.goToStep(0);
            portal.checkLocalAlerts();
            portal.prefetchCriticalData();
            
            console.log("✅ Professional portal ready for emergencies");
        } catch (error) {
            console.error("❌ Portal initialization failed:", error);
            portal.showFallbackInterface();
        }
    }

    // --- SMART EMERGENCY SHORTCUTS ---
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Emergency shortcuts (work from any step)
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'e': // Ctrl+E = Emergency report
                        e.preventDefault();
                        portal.quickEmergency();
                        break;
                    case 'v': // Ctrl+V = Voice report
                        e.preventDefault();
                        portal.goToStep(3);
                        break;
                    case 's': // Ctrl+S = Save current progress
                        e.preventDefault();
                        portal.saveProgress();
                        break;
                }
            }
            
            // Step navigation (arrow keys)
            if (!e.target.matches('input, textarea, select')) {
                switch(e.key) {
                    case 'ArrowRight':
                        if (portal.currentStep < 4) portal.goToStep(portal.currentStep + 1);
                        break;
                    case 'ArrowLeft':
                        if (portal.currentStep > 0) portal.goToStep(portal.currentStep - 1);
                        break;
                    case 'Escape':
                        portal.handleEscapeKey();
                        break;
                    case 'Enter':
                        if (e.target.matches('button')) {
                            e.target.click();
                        }
                        break;
                }
            }
        });
    }

    // --- ENHANCED NAVIGATION WITH UX IMPROVEMENTS ---
    portal.goToStep = function(stepNumber, skipValidation = false) {
        try {
            if (typeof stepNumber !== 'number' || stepNumber < 0 || stepNumber > 4) return;
            
            // Track timing for UX optimization
            const startTime = Date.now();
            portal.stepTimes[portal.currentStep] = startTime - (portal.stepTimes.lastStart || startTime);
            portal.stepTimes.lastStart = startTime;
            
            // Validate current step before leaving (unless skipping)
            if (!skipValidation && !portal.validateCurrentStep()) {
                return false;
            }
            
            // Save current progress
            portal.saveProgress();
            
            // Update UI with smooth animations
            const sections = document.querySelectorAll('.content-section');
            const currentSection = document.getElementById(`step-${portal.currentStep}`);
            const targetSection = document.getElementById(`step-${stepNumber}`);
            
            // Animate out current section
            if (currentSection) {
                currentSection.style.transform = stepNumber > portal.currentStep ? 'translateX(-100%)' : 'translateX(100%)';
                currentSection.style.opacity = '0';
                
                setTimeout(() => {
                    sections.forEach(s => s.classList.remove('active'));
                    if (targetSection) {
                        targetSection.classList.add('active');
                        targetSection.style.transform = stepNumber > portal.currentStep ? 'translateX(100%)' : 'translateX(-100%)';
                        targetSection.style.opacity = '0';
                        
                        // Animate in target section
                        requestAnimationFrame(() => {
                            targetSection.style.transition = 'all 0.3s ease';
                            targetSection.style.transform = 'translateX(0)';
                            targetSection.style.opacity = '1';
                        });
                    }
                }, 150);
            } else {
                // First load - no animation
                sections.forEach(s => s.classList.remove('active'));
                if (targetSection) {
                    targetSection.classList.add('active');
                    targetSection.style.opacity = '1';
                    targetSection.style.transform = 'translateX(0)';
                }
            }
            
            // Update state and indicators
            const previousStep = portal.currentStep;
            portal.currentStep = stepNumber;
            updateStepIndicator(stepNumber, previousStep);
            
            // Initialize step-specific functionality
            setTimeout(() => {
                switch(stepNumber) {
                    case 0: portal.initializeWelcome(); break;
                    case 1: portal.initializeSetup(); break;
                    case 2: portal.initializeReportForm(); break;
                    case 3: portal.initializeVoiceReporter(); break;
                    case 4: portal.loadUserReports(); break;
                }
                
                // Focus management for accessibility
                portal.manageFocus(stepNumber);
                
                // Analytics tracking
                portal.trackStepChange(previousStep, stepNumber);
            }, 300);
            
            console.log(`📍 Navigated to step ${stepNumber} (${portal.getStepName(stepNumber)})`);
            return true;
            
        } catch (error) {
            console.error("❌ Error navigating to step:", error);
            return false;
        }
    };

    // --- ENHANCED STEP VALIDATION ---
    portal.validateCurrentStep = function() {
        const currentStep = portal.currentStep;
        let isValid = true;
        let message = '';
        
        switch(currentStep) {
            case 1: // Setup validation
                if (!portal.permissionsGranted.location && portal.formData.reportMethod !== 'text') {
                    isValid = false;
                    message = '📍 Location access is recommended for emergency reporting';
                }
                break;
                
            case 2: // Report form validation
                if (!portal.formData.description.trim()) {
                    isValid = false;
                    message = '📝 Please describe the emergency situation';
                } else if (portal.formData.description.length < 10) {
                    isValid = false;
                    message = '📝 Please provide more details about the emergency';
                }
                break;
                
            case 3: // Voice validation
                if (!portal.permissionsGranted.microphone) {
                    isValid = false;
                    message = '🎤 Microphone access is required for voice reporting';
                }
                break;
        }
        
        if (!isValid && message) {
            portal.showValidationMessage(message);
        }
        
        return isValid;
    };

    portal.showValidationMessage = function(message) {
        const existingMessage = document.querySelector('.validation-message');
        if (existingMessage) existingMessage.remove();
        
        const messageEl = document.createElement('div');
        messageEl.className = 'validation-message';
        messageEl.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #dc2626;
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 10001;
            font-weight: 600;
            text-align: center;
            max-width: 400px;
            animation: shakeIn 0.5s ease;
        `;
        messageEl.textContent = message;
        
        document.body.appendChild(messageEl);
        
        setTimeout(() => {
            messageEl.style.animation = 'fadeOut 0.3s ease forwards';
            setTimeout(() => messageEl.remove(), 300);
        }, 3000);
    };

    // --- ENHANCED FORM SYSTEM ---
    portal.initializeReportForm = function() {
        console.log("📝 Initializing enhanced report form...");
        
        const formContainer = document.getElementById('emergencyReportForm');
        if (!formContainer) return;
        
        // Clear previous form
        formContainer.innerHTML = '';
        
        // Build form based on selected method
        const formHTML = portal.buildDynamicForm();
        formContainer.innerHTML = formHTML;
        
        // Setup form interactions
        portal.setupFormValidation();
        portal.setupAutoComplete();
        portal.setupSmartDefaults();
        portal.loadPreviousData();
        
        // Focus first input for accessibility
        const firstInput = formContainer.querySelector('input, textarea, select');
        if (firstInput) {
            setTimeout(() => firstInput.focus(), 100);
        }
    };

    portal.buildDynamicForm = function() {
        const method = portal.formData.reportMethod;
        
        let formHTML = `
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label required">📋 Emergency Type</label>
                    <select class="form-input" id="emergencyType" required>
                        <option value="">Select emergency type...</option>
                        <option value="fire">🔥 Fire</option>
                        <option value="medical">🏥 Medical Emergency</option>
                        <option value="accident">🚗 Traffic Accident</option>
                        <option value="crime">👮 Crime in Progress</option>
                        <option value="weather">🌪️ Severe Weather</option>
                        <option value="hazmat">☢️ Hazardous Materials</option>
                        <option value="infrastructure">🏗️ Infrastructure Failure</option>
                        <option value="other">❓ Other Emergency</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label required">⚡ Priority Level</label>
                    <select class="form-input" id="priorityLevel" required>
                        <option value="critical">🔴 Critical - Life Threatening</option>
                        <option value="high">🟠 High - Urgent Response Needed</option>
                        <option value="medium" selected>🟡 Medium - Timely Response</option>
                        <option value="low">🟢 Low - Non-urgent</option>
                    </select>
                </div>
            </div>
            
            <div class="form-group">
                <label class="form-label required">📝 Describe the Emergency</label>
                <textarea 
                    class="form-input form-textarea" 
                    id="emergencyDescription" 
                    placeholder="Describe what's happening, who is involved, and any immediate dangers..."
                    required
                    minlength="10"
                    maxlength="1000"
                ></textarea>
                <div class="char-counter">
                    <span id="charCount">0</span>/1000 characters
                </div>
            </div>
            
            <div class="form-group">
                <label class="form-label">📍 Location Details</label>
                <input 
                    type="text" 
                    class="form-input" 
                    id="emergencyLocation" 
                    placeholder="Address, intersection, or landmark..."
                    value="${portal.userLocation ? 'Using GPS location' : ''}"
                >
                <button type="button" class="btn btn-secondary btn-sm" onclick="portal.useCurrentLocation()">
                    📍 Use My Location
                </button>
            </div>
        `;
        
        // Add method-specific fields
        if (method === 'photo') {
            formHTML += `
                <div class="form-group">
                    <label class="form-label">📸 Photo Evidence</label>
                    <div class="file-upload-area" onclick="document.getElementById('photoUpload').click()">
                        <div class="upload-placeholder">
                            <span class="upload-icon">📷</span>
                            <span class="upload-text">Tap to take photo or upload image</span>
                            <span class="upload-hint">Supports: JPG, PNG, HEIC (max 10MB)</span>
                        </div>
                    </div>
                    <input type="file" id="photoUpload" accept="image/*" capture="environment" style="display: none;">
                </div>
            `;
        }
        
        // Add contact information section
        formHTML += `
            <div class="form-group">
                <label class="form-label">📞 Contact Information (Optional)</label>
                <div class="form-row">
                    <input 
                        type="text" 
                        class="form-input" 
                        id="reporterName" 
                        placeholder="Your name"
                    >
                    <input 
                        type="tel" 
                        class="form-input" 
                        id="reporterPhone" 
                        placeholder="Phone number"
                    >
                </div>
                <p class="form-text">This helps responders contact you for updates or additional information.</p>
            </div>
            
            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="consentToContact" checked>
                    <span class="checkmark"></span>
                    I consent to be contacted by emergency responders about this report
                </label>
            </div>
        `;
        
        return formHTML;
    };

    // --- REAL-TIME FORM ENHANCEMENTS ---
    portal.setupFormValidation = function() {
        const form = document.getElementById('emergencyReportForm');
        if (!form) return;
        
        // Real-time validation
        const inputs = form.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            input.addEventListener('input', portal.handleInputChange);
            input.addEventListener('blur', portal.validateField);
        });
        
        // Character counter for description
        const description = document.getElementById('emergencyDescription');
        const charCount = document.getElementById('charCount');
        if (description && charCount) {
            description.addEventListener('input', (e) => {
                const count = e.target.value.length;
                charCount.textContent = count;
                charCount.parentElement.className = `char-counter ${count > 800 ? 'warning' : ''} ${count >= 1000 ? 'error' : ''}`;
            });
        }
        
        // File upload handling
        const fileInput = document.getElementById('photoUpload');
        if (fileInput) {
            fileInput.addEventListener('change', portal.handleFileUpload);
        }
    };

    portal.handleInputChange = function(e) {
        const field = e.target;
        portal.formData[field.id] = field.value;
        portal.hasUnsavedChanges = true;
        
        // Auto-save after short delay
        clearTimeout(portal.autoSaveTimeout);
        portal.autoSaveTimeout = setTimeout(() => {
            portal.saveProgress();
        }, 2000);
        
        // Update form validity
        portal.updateFormValidity();
    };

    portal.validateField = function(e) {
        const field = e.target;
        const value = field.value.trim();
        
        // Remove existing validation styling
        field.classList.remove('valid', 'invalid');
        
        let isValid = true;
        let message = '';
        
        if (field.required && !value) {
            isValid = false;
            message = 'This field is required';
        } else if (field.id === 'emergencyDescription' && value.length < 10) {
            isValid = false;
            message = 'Please provide more details (at least 10 characters)';
        } else if (field.type === 'tel' && value && !portal.isValidPhone(value)) {
            isValid = false;
            message = 'Please enter a valid phone number';
        }
        
        // Apply validation styling
        field.classList.add(isValid ? 'valid' : 'invalid');
        
        // Show/hide validation message
        let msgEl = field.parentElement.querySelector('.field-validation');
        if (!isValid && message) {
            if (!msgEl) {
                msgEl = document.createElement('div');
                msgEl.className = 'field-validation';
                field.parentElement.appendChild(msgEl);
            }
            msgEl.textContent = message;
            msgEl.style.display = 'block';
        } else if (msgEl) {
            msgEl.style.display = 'none';
        }
        
        return isValid;
    };

    // --- ENHANCED LOCATION SERVICES ---
    portal.useCurrentLocation = function() {
        const locationInput = document.getElementById('emergencyLocation');
        if (!locationInput) return;
        
        locationInput.value = 'Getting location...';
        locationInput.disabled = true;
        
        if (!navigator.geolocation) {
            locationInput.value = 'Location not available';
            locationInput.disabled = false;
            return;
        }
        
        const options = {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 60000
        };
        
        navigator.geolocation.getCurrentPosition(
            async (position) => {
                portal.userLocation = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude,
                    accuracy: position.coords.accuracy
                };
                
                // Reverse geocoding for human-readable address
                try {
                    const address = await portal.reverseGeocode(portal.userLocation);
                    locationInput.value = address;
                    portal.showNotification('📍 Location updated', 'success');
                } catch (error) {
                    locationInput.value = `${portal.userLocation.lat.toFixed(6)}, ${portal.userLocation.lng.toFixed(6)}`;
                    portal.showNotification('📍 GPS coordinates captured', 'info');
                }
                
                locationInput.disabled = false;
            },
            (error) => {
                console.error('Location error:', error);
                locationInput.value = '';
                locationInput.disabled = false;
                portal.showNotification('❌ Could not get location', 'error');
            },
            options
        );
    };

    portal.reverseGeocode = async function(location) {
        // In a real app, you'd use a geocoding service like Google Maps API
        // For now, return coordinates
        return `${location.lat.toFixed(6)}, ${location.lng.toFixed(6)}`;
    };

    // --- ENHANCED VOICE RECORDING ---
    portal.setupAdvancedSpeechRecognition = function() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.log('Speech recognition not supported');
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        portal.recognition = new SpeechRecognition();
        
        portal.recognition.continuous = true;
        portal.recognition.interimResults = true;
        portal.recognition.lang = 'en-US';
        
        portal.recognition.onstart = () => {
            portal.showNotification('🎤 Listening...', 'info');
        };
        
        portal.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            const transcriptBox = document.getElementById('transcriptBox');
            if (transcriptBox) {
                portal.voiceTranscript = finalTranscript;
                transcriptBox.innerHTML = `
                    <div class="final-transcript">${finalTranscript}</div>
                    <div class="interim-transcript">${interimTranscript}</div>
                `;
                
                // Auto-analysis for keywords
                if (finalTranscript) {
                    portal.analyzeVoiceContent(finalTranscript);
                }
            }
        };
        
        portal.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            portal.showNotification('🎤 Voice recognition error', 'error');
        };
        
        portal.recognition.onend = () => {
            portal.isVoiceRecording = false;
            portal.updateVoiceUI();
        };
    };

    portal.analyzeVoiceContent = function(transcript) {
        const urgencyKeywords = {
            critical: ['fire', 'burning', 'explosion', 'heart attack', 'stroke', 'bleeding', 'unconscious', 'trapped', 'help me'],
            high: ['accident', 'injured', 'emergency', 'urgent', 'police', 'ambulance', 'broken'],
            medium: ['problem', 'issue', 'concerned', 'hurt', 'damage'],
            low: ['question', 'information', 'minor', 'small']
        };
        
        const emotionKeywords = {
            panic: ['help', 'scared', 'terrified', 'panic', 'dying'],
            concerned: ['worried', 'concerned', 'anxious', 'afraid'],
            calm: ['reporting', 'witnessed', 'observed', 'information']
        };
        
        let detectedUrgency = 'medium';
        let detectedEmotion = 'calm';
        
        const lowerTranscript = transcript.toLowerCase();
        
        // Analyze urgency
        for (const [level, keywords] of Object.entries(urgencyKeywords)) {
            if (keywords.some(keyword => lowerTranscript.includes(keyword))) {
                detectedUrgency = level;
                break;
            }
        }
        
        // Analyze emotion
        for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
            if (keywords.some(keyword => lowerTranscript.includes(keyword))) {
                detectedEmotion = emotion;
                break;
            }
        }
        
        // Update UI with analysis
        portal.updateVoiceAnalysis({
            urgency: detectedUrgency,
            emotion: detectedEmotion,
            transcript: transcript,
            confidence: 0.8
        });
    };

    portal.updateVoiceAnalysis = function(analysis) {
        const analysisDiv = document.getElementById('voiceAnalysis');
        if (!analysisDiv) return;
        
        // Show analysis results
        analysisDiv.style.display = 'block';
        analysisDiv.classList.add('visible');
        
        // Update individual fields
        const urgencyEl = document.getElementById('urgencyLevel');
        const emotionEl = document.getElementById('emotionLevel');
        const locationEl = document.getElementById('detectedLocation');
        const recommendationEl = document.getElementById('aiRecommendation');
        
        if (urgencyEl) {
            urgencyEl.textContent = portal.getUrgencyText(analysis.urgency);
            urgencyEl.className = `urgency-${analysis.urgency}`;
        }
        
        if (emotionEl) {
            emotionEl.textContent = portal.getEmotionText(analysis.emotion);
            emotionEl.className = `emotion-${analysis.emotion}`;
        }
        
        if (locationEl) {
            locationEl.textContent = portal.userLocation ? 
                'GPS location available' : 
                'No location detected';
        }
        
        if (recommendationEl) {
            recommendationEl.textContent = portal.getRecommendation(analysis);
        }
        
        // Enable submit button
        const submitBtn = document.getElementById('submitVoiceReport');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.classList.add('ready');
        }
    };

    // --- UTILITY FUNCTIONS ---
    portal.getStepName = function(stepNumber) {
        const names = ['Welcome', 'Setup', 'Report', 'Voice', 'Track'];
        return names[stepNumber] || 'Unknown';
    };

    portal.getUrgencyText = function(urgency) {
        const texts = {
            critical: '🔴 Critical - Immediate response needed',
            high: '🟠 High - Urgent response required',
            medium: '🟡 Medium - Timely response needed',
            low: '🟢 Low - Non-urgent situation'
        };
        return texts[urgency] || texts.medium;
    };

    portal.getEmotionText = function(emotion) {
        const texts = {
            panic: '😰 High stress detected',
            concerned: '😟 Moderate concern',
            calm: '😌 Calm and composed'
        };
        return texts[emotion] || texts.calm;
    };

    portal.getRecommendation = function(analysis) {
        if (analysis.urgency === 'critical') {
            return '🚨 Submit immediately - Emergency services will be dispatched';
        } else if (analysis.urgency === 'high') {
            return '⚡ Submit soon - Urgent response team will be notified';
        } else {
            return '📋 Review details and submit when ready';
        }
    };

    portal.isValidPhone = function(phone) {
        const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
        return phoneRegex.test(phone.replace(/[\s\-\(\)]/g, ''));
    };

    // --- ENHANCED AUTO-SAVE AND PROGRESS ---
    portal.saveProgress = function() {
        const progressData = {
            currentStep: portal.currentStep,
            formData: portal.formData,
            timestamp: Date.now(),
            voiceTranscript: portal.voiceTranscript,
            userLocation: portal.userLocation
        };
        
        try {
            localStorage.setItem('emergencyPortalProgress', JSON.stringify(progressData));
            portal.hasUnsavedChanges = false;
            
            // Show subtle save indicator
            portal.showSaveIndicator();
        } catch (error) {
            console.error('Failed to save progress:', error);
        }
    };

    portal.loadProgress = function() {
        try {
            const saved = localStorage.getItem('emergencyPortalProgress');
            if (saved) {
                const progressData = JSON.parse(saved);
                
                // Only restore if recent (within 24 hours)
                if (Date.now() - progressData.timestamp < 24 * 60 * 60 * 1000) {
                    Object.assign(portal.formData, progressData.formData);
                    portal.voiceTranscript = progressData.voiceTranscript || '';
                    portal.userLocation = progressData.userLocation || null;
                    
                    return progressData.currentStep || 0;
                }
            }
        } catch (error) {
            console.error('Failed to load progress:', error);
        }
        return 0;
    };

    portal.showSaveIndicator = function() {
        const indicator = document.createElement('div');
        indicator.className = 'save-indicator';
        indicator.textContent = '💾 Saved';
        indicator.style.cssText = `
            position: fixed;
            top: 20px;
            left: 20px;
            background: #16a34a;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            z-index: 9999;
            opacity: 0;
            animation: saveIndicator 2s ease;
        `;
        
        document.body.appendChild(indicator);
        setTimeout(() => indicator.remove(), 2000);
    };

    // --- ACCESSIBILITY ENHANCEMENTS ---
    portal.manageFocus = function(stepNumber) {
        const stepElement = document.getElementById(`step-${stepNumber}`);
        if (!stepElement) return;
        
        // Find the best element to focus
        const focusTargets = [
            'input:not([disabled])',
            'textarea:not([disabled])',
            'select:not([disabled])',
            'button:not([disabled])',
            '[tabindex="0"]'
        ];
        
        for (const selector of focusTargets) {
            const target = stepElement.querySelector(selector);
            if (target) {
                setTimeout(() => {
                    target.focus();
                    // Scroll into view if needed
                    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 400);
                break;
            }
        }
    };

    // Re-implement essential functions with enhancements
    portal.selectMethod = function(element, method) {
        const cards = document.querySelectorAll('.method-card');
        cards.forEach(c => c.classList.remove('selected'));
        element.classList.add('selected');
        
        portal.formData.reportMethod = method;
        console.log(`Selected method: ${method}`);
        
        // Rebuild form for selected method
        setTimeout(() => {
            portal.initializeReportForm();
        }, 100);
        
        portal.showNotification(`📱 ${method} reporting method selected`, 'info');
    };

    portal.submitReport = function() {
        // Validate form before submission
        if (!portal.validateCurrentStep()) {
            return;
        }
        
        // Collect all form data
        const reportData = {
            type: document.getElementById('emergencyType')?.value || 'other',
            priority: document.getElementById('priorityLevel')?.value || 'medium',
            description: document.getElementById('emergencyDescription')?.value || '',
            location: document.getElementById('emergencyLocation')?.value || '',
            method: portal.formData.reportMethod,
            reporterName: document.getElementById('reporterName')?.value || '',
            reporterPhone: document.getElementById('reporterPhone')?.value || '',
            consent: document.getElementById('consentToContact')?.checked || false,
            coordinates: portal.userLocation,
            timestamp: new Date().toISOString(),
            reportId: `ER-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        };
        
        // Save to local storage
        const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
        savedReports.push(reportData);
        localStorage.setItem('userReports', JSON.stringify(savedReports));
        
        // Submit to server (with offline fallback)
        portal.submitToServer(reportData);
        
        // Clear saved progress
        localStorage.removeItem('emergencyPortalProgress');
        
        // Show success and navigate
        portal.showNotification('✅ Emergency report submitted successfully!', 'success');
        setTimeout(() => portal.goToStep(4), 1500);
    };

    portal.submitToServer = async function(reportData) {
        try {
            const response = await fetch('/api/submit-emergency-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(reportData)
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Report submitted to server:', result);
            } else {
                throw new Error('Server submission failed');
            }
        } catch (error) {
            console.error('Failed to submit to server:', error);
            // Add to offline queue
            const offlineQueue = JSON.parse(localStorage.getItem('offlineQueue') || '[]');
            offlineQueue.push(reportData);
            localStorage.setItem('offlineQueue', JSON.stringify(offlineQueue));
            
            portal.showNotification('📴 Saved offline - will sync when connected', 'warning');
        }
    };

    portal.submitVoiceReport = function() {
        if (!portal.voiceTranscript) {
            portal.showNotification('🎤 No voice recording found', 'error');
            return;
        }
        
        const voiceData = {
            transcript: portal.voiceTranscript,
            analysis: portal.currentAnalysis,
            duration: portal.recordingDuration || 0,
            timestamp: new Date().toISOString(),
            location: portal.userLocation,
            reportId: `VR-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        };
        
        // Save locally
        const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
        savedReports.push(voiceData);
        localStorage.setItem('userReports', JSON.stringify(savedReports));
        
        // Submit to server
        portal.submitVoiceToServer(voiceData);
        
        portal.showNotification('🎤 Voice report submitted successfully!', 'success');
        setTimeout(() => portal.goToStep(4), 1500);
    };

    portal.submitVoiceToServer = async function(voiceData) {
        try {
            const response = await fetch('/api/submit-voice-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(voiceData)
            });
            
            if (response.ok) {
                console.log('Voice report submitted successfully');
            }
        } catch (error) {
            console.error('Failed to submit voice report:', error);
        }
    };

    // --- ENHANCED VOICE CONTROLS ---
    portal.toggleRecording = function() {
        if (portal.isVoiceRecording) {
            portal.stopVoiceRecording();
        } else {
            portal.startVoiceRecording();
        }
    };

    portal.startVoiceRecording = function() {
        if (!portal.permissionsGranted.microphone) {
            portal.showNotification('🎤 Please enable microphone access first', 'warning');
            portal.goToStep(1); // Go to setup
            return;
        }

        portal.isVoiceRecording = true;
        portal.recordingStartTime = Date.now();
        
        // Update UI
        portal.updateVoiceUI();
        
        // Start speech recognition
        if (portal.recognition) {
            portal.recognition.start();
        }
        
        // Start audio recording for backup
        portal.startAudioRecording();
        
        portal.showNotification('🎤 Voice recording started - speak clearly', 'success');
    };

    portal.stopVoiceRecording = function() {
        portal.isVoiceRecording = false;
        portal.recordingDuration = Date.now() - (portal.recordingStartTime || Date.now());
        
        // Stop speech recognition
        if (portal.recognition) {
            portal.recognition.stop();
        }
        
        // Stop audio recording
        portal.stopAudioRecording();
        
        // Update UI
        portal.updateVoiceUI();
        
        // Enable submit if we have content
        if (portal.voiceTranscript) {
            const submitBtn = document.getElementById('submitVoiceReport');
            if (submitBtn) {
                submitBtn.disabled = false;
            }
        }
        
        portal.showNotification('🎤 Recording stopped - processing...', 'info');
    };

    portal.updateVoiceUI = function() {
        const recordButton = document.getElementById('recordButton');
        const voiceStatus = document.getElementById('voiceStatus');
        
        if (recordButton) {
            if (portal.isVoiceRecording) {
                recordButton.innerHTML = '⏹️';
                recordButton.style.background = '#dc2626';
                recordButton.classList.add('recording');
            } else {
                recordButton.innerHTML = '🎤';
                recordButton.style.background = '';
                recordButton.classList.remove('recording');
            }
        }
        
        if (voiceStatus) {
            if (portal.isVoiceRecording) {
                voiceStatus.textContent = '🎤 Recording... Speak clearly about the emergency';
            } else if (portal.voiceTranscript) {
                voiceStatus.textContent = '✅ Recording complete. Review and submit.';
            } else {
                voiceStatus.textContent = 'Tap the microphone to start recording';
            }
        }
    };

    portal.startAudioRecording = function() {
        if (!navigator.mediaDevices) return;
        
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                portal.mediaRecorder = new MediaRecorder(stream);
                portal.audioChunks = [];
                
                portal.mediaRecorder.ondataavailable = event => {
                    portal.audioChunks.push(event.data);
                };
                
                portal.mediaRecorder.start();
            })
            .catch(error => {
                console.error('Audio recording failed:', error);
            });
    };

    portal.stopAudioRecording = function() {
        if (portal.mediaRecorder && portal.mediaRecorder.state !== 'inactive') {
            portal.mediaRecorder.stop();
            portal.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    };

    // --- PERMISSIONS WITH ENHANCED UX ---
    portal.requestLocation = async function(button) {
        if (!navigator.geolocation) {
            portal.showNotification('📍 Geolocation is not supported on this device', 'error');
            return;
        }
        
        button.textContent = 'Requesting...';
        button.disabled = true;
        
        try {
            const pos = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, { 
                    timeout: 15000,
                    enableHighAccuracy: true,
                    maximumAge: 300000
                });
            });
            
            portal.userLocation = { 
                lat: pos.coords.latitude, 
                lng: pos.coords.longitude,
                accuracy: pos.coords.accuracy
            };
            portal.permissionsGranted.location = true;
            
            const statusEl = document.getElementById('locationStatus');
            if (statusEl) {
                statusEl.innerHTML = `✅ Location: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)} <br><small>Accuracy: ±${Math.round(pos.coords.accuracy)}m</small>`;
            }
            
            portal.showPermissionSuccess(button, 'Location');
            portal.showNotification('📍 Location access granted', 'success');
            
        } catch (err) {
            console.error('Location error:', err);
            portal.showPermissionError(button, 'Location');
            
            const statusEl = document.getElementById('locationStatus');
            if (statusEl) {
                statusEl.innerHTML = '❌ Location access denied or unavailable<br><small>You can still report by entering your address manually</small>';
            }
            
            let errorMsg = 'Location access denied';
            if (err.code === 1) errorMsg = 'Location access denied by user';
            else if (err.code === 2) errorMsg = 'Location unavailable';
            else if (err.code === 3) errorMsg = 'Location request timed out';
            
            portal.showNotification(`📍 ${errorMsg}`, 'warning');
        }
    };
    
    portal.requestMicrophone = async function(btn) {
        try {
            btn.textContent = 'Requesting...';
            btn.disabled = true;
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Test recording briefly
            const testRecorder = new MediaRecorder(stream);
            testRecorder.start();
            setTimeout(() => {
                testRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }, 100);
            
            portal.permissionsGranted.microphone = true;
            portal.showPermissionSuccess(btn, 'Microphone');
            portal.showNotification('🎤 Microphone access granted', 'success');
            
        } catch (err) {
            console.error('Microphone error:', err);
            portal.showPermissionError(btn, 'Microphone');
            portal.showNotification('🎤 Microphone access denied', 'error');
        }
    };
    
    portal.requestCamera = async function(btn) {
        try {
            btn.textContent = 'Requesting...';
            btn.disabled = true;
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            });
            
            // Stop the stream immediately after permission granted
            stream.getTracks().forEach(track => track.stop());
            
            portal.permissionsGranted.camera = true;
            portal.showPermissionSuccess(btn, 'Camera');
            portal.showNotification('📷 Camera access granted', 'success');
            
        } catch (err) {
            console.error('Camera error:', err);
            portal.showPermissionError(btn, 'Camera');
            portal.showNotification('📷 Camera access denied', 'error');
        }
    };

    portal.requestNotifications = async function(btn) {
        if (!('Notification' in window)) {
            portal.showPermissionError(btn, 'Notifications');
            portal.showNotification('🔔 Notifications not supported', 'error');
            return;
        }
        
        try {
            btn.textContent = 'Requesting...';
            btn.disabled = true;
            
            const permission = await Notification.requestPermission();
            if (permission === 'granted') {
                portal.permissionsGranted.notifications = true;
                portal.showPermissionSuccess(btn, 'Notifications');
                
                // Show test notification
                new Notification('Emergency Assistant', { 
                    body: '🔔 Emergency alerts are now enabled!',
                    icon: '/static/icons/notification-icon.png',
                    badge: '/static/icons/badge-icon.png',
                    tag: 'emergency-test'
                });
                
                portal.showNotification('🔔 Notifications enabled', 'success');
            } else {
                portal.showPermissionError(btn, 'Notifications');
                portal.showNotification('🔔 Notification permission denied', 'warning');
            }
        } catch (err) {
            console.error('Notification error:', err);
            portal.showPermissionError(btn, 'Notifications');
        }
    };

    // --- STEP-SPECIFIC INITIALIZATIONS ---
    portal.initializeWelcome = function() {
        console.log("🏠 Welcome step initialized");
        
        // Load saved progress if available
        const savedStep = portal.loadProgress();
        if (savedStep > 0) {
            portal.showContinueOption(savedStep);
        }
        
        // Setup feature card interactions
        portal.setupFeatureCards();
    };

    portal.showContinueOption = function(savedStep) {
        const continueDiv = document.createElement('div');
        continueDiv.className = 'continue-option';
        continueDiv.innerHTML = `
            <div class="continue-card">
                <h3>📄 Continue Previous Session</h3>
                <p>You have unsaved progress from step ${portal.getStepName(savedStep)}.</p>
                <div class="continue-buttons">
                    <button class="btn btn-primary" onclick="portal.goToStep(${savedStep})">
                        Continue from ${portal.getStepName(savedStep)}
                    </button>
                    <button class="btn btn-secondary" onclick="portal.clearProgress()">
                        Start Fresh
                    </button>
                </div>
            </div>
        `;
        
        const welcomeSection = document.getElementById('step-0');
        if (welcomeSection) {
            welcomeSection.insertBefore(continueDiv, welcomeSection.querySelector('.features-grid'));
        }
    };

    portal.clearProgress = function() {
        localStorage.removeItem('emergencyPortalProgress');
        portal.formData = {
            reportMethod: 'text',
            priority: 'medium',
            description: '',
            location: '',
            evidence: null,
            autoSave: true
        };
        portal.voiceTranscript = '';
        
        // Remove continue option
        const continueOption = document.querySelector('.continue-option');
        if (continueOption) continueOption.remove();
        
        portal.showNotification('🔄 Started fresh session', 'info');
    };

    portal.initializeSetup = function() {
        console.log("🛠️ Setup step initialized");
        portal.checkExistingPermissions();
        
        // Show permission importance based on selected method
        portal.updatePermissionPriority();
    };

    portal.updatePermissionPriority = function() {
        const method = portal.formData.reportMethod;
        
        // Update UI to show which permissions are most important
        const permissionGroups = document.querySelectorAll('.form-group');
        permissionGroups.forEach(group => {
            group.classList.remove('priority-high', 'priority-medium', 'priority-low');
        });
        
        // Set priorities based on method
        if (method === 'voice') {
            document.querySelector('label[for*="microphone"]')?.parentElement.classList.add('priority-high');
            document.querySelector('label[for*="location"]')?.parentElement.classList.add('priority-medium');
        } else if (method === 'photo') {
            document.querySelector('label[for*="camera"]')?.parentElement.classList.add('priority-high');
            document.querySelector('label[for*="location"]')?.parentElement.classList.add('priority-high');
        } else {
            document.querySelector('label[for*="location"]')?.parentElement.classList.add('priority-high');
        }
    };

    portal.initializeVoiceReporter = function() {
        console.log("🎤 Voice reporter initialized");
        
        const transcriptBox = document.getElementById('transcriptBox');
        const voiceStatus = document.getElementById('voiceStatus');
        const submitBtn = document.getElementById('submitVoiceReport');
        
        if (transcriptBox) {
            transcriptBox.innerHTML = portal.voiceTranscript || 
                '<div class="transcript-placeholder">Your voice transcript will appear here...</div>';
        }
        
        if (voiceStatus) {
            voiceStatus.textContent = portal.voiceTranscript ? 
                '✅ Recording available. Review and submit.' :
                'Tap the microphone to start recording';
        }
        
        if (submitBtn) {
            submitBtn.disabled = !portal.voiceTranscript;
        }
        
        // Check microphone permission
        if (!portal.permissionsGranted.microphone) {
            portal.showMicrophoneSetupPrompt();
        }
    };

    portal.showMicrophoneSetupPrompt = function() {
        const promptDiv = document.createElement('div');
        promptDiv.className = 'permission-prompt';
        promptDiv.innerHTML = `
            <div class="prompt-card">
                <h3>🎤 Microphone Access Required</h3>
                <p>Voice reporting requires microphone access to record your emergency report.</p>
                <button class="btn btn-primary" onclick="portal.requestMicrophone(this); this.parentElement.parentElement.remove();">
                    Enable Microphone
                </button>
                <button class="btn btn-secondary" onclick="portal.goToStep(2)">
                    Use Text Instead
                </button>
            </div>
        `;
        
        const voiceSection = document.getElementById('step-3');
        if (voiceSection) {
            voiceSection.insertBefore(promptDiv, voiceSection.querySelector('.voice-interface'));
        }
    };

    // --- ESSENTIAL SETUP FUNCTIONS ---
    portal.checkExistingPermissions = function() {
        console.log("🔍 Checking existing permissions...");
        
        // Check geolocation permission and get location if available
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                pos => {
                    portal.userLocation = { 
                        lat: pos.coords.latitude, 
                        lng: pos.coords.longitude,
                        accuracy: pos.coords.accuracy
                    };
                    portal.permissionsGranted.location = true;
                    
                    const statusEl = document.getElementById('locationStatus');
                    if (statusEl) {
                        statusEl.innerHTML = `✅ Location: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`;
                    }
                    
                    const btn = document.querySelector('button[onclick*="requestLocation"]');
                    if (btn) portal.showPermissionSuccess(btn, 'Location');
                }, 
                () => {}, 
                { timeout: 5000, maximumAge: 300000 }
            );
        }

        // Check notification permission
        if ('Notification' in window && Notification.permission === 'granted') {
            portal.permissionsGranted.notifications = true;
            const btn = document.querySelector('button[onclick*="requestNotifications"]');
            if (btn) portal.showPermissionSuccess(btn, 'Notifications');
        }
        
        // Check media permissions
        if (navigator.mediaDevices) {
            navigator.mediaDevices.enumerateDevices().then(devices => {
                const hasAudio = devices.some(device => device.kind === 'audioinput');
                const hasVideo = devices.some(device => device.kind === 'videoinput');
                
                if (!hasAudio) {
                    console.log('No microphone detected');
                }
                if (!hasVideo) {
                    console.log('No camera detected');
                }
            });
        }
    };

    // Initialize CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes shakeIn {
            0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0; }
            50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
            100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }
        
        @keyframes fadeOut {
            to { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
        }
        
        @keyframes saveIndicator {
            0% { opacity: 0; transform: translateX(-20px); }
            20% { opacity: 1; transform: translateX(0); }
            80% { opacity: 1; transform: translateX(0); }
            100% { opacity: 0; transform: translateX(-20px); }
        }
        
        .content-section {
            transition: all 0.3s ease;
        }
    `;
    document.head.appendChild(style);

    // --- MAKE PORTAL GLOBALLY ACCESSIBLE ---
    window.portal = portal;
    
    // --- INITIALIZE THE APPLICATION ---
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeApp);
    } else {
        initializeApp();
    }

    console.log("✅ Professional Emergency Portal loaded successfully");

})();