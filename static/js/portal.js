/**
 * portal.js - Enhanced Citizen Portal JavaScript
 * -------------------------------------------------
 * This script powers the multi-step emergency reporting interface in home.html.
 * It handles UI navigation, permissions, form submissions, offline queueing,
 * and simulates real-time AI analysis for a dynamic user experience.
 */

(function() {
    "use strict";

    // --- GLOBAL PORTAL OBJECT & STATE ---
    const portal = {
        currentStep: 0,
        userLocation: null,
        isVoiceRecording: false, // For the main voice reporter
        isAudioNoteRecording: false, // For the form's audio note
        mediaRecorder: null,
        audioChunks: [],
        recognition: null,
        permissionsGranted: { location: false, microphone: false, camera: false, notifications: false },
        lastEscapeTime: null,
        currentAnalysis: {},
        voiceTranscript: '',
        currentReportData: {},
        userReports: []
    };

    // --- INITIALIZATION ---
    function initializeApp() {
        console.log("🚀 Portal Initializing...");
        setupEventListeners();
        portal.checkExistingPermissions();
        portal.setupMethodSelector();
        portal.setupSpeechRecognition();
        portal.updateAIStatus('ready');
        portal.setupOfflineHandling();
        portal.goToStep(0);
        portal.checkLocalAlerts();
        console.log("✅ Portal initialization complete.");
    }

    function setupEventListeners() {
        document.querySelectorAll('.nav-btn').forEach(btn => {
            const href = btn.getAttribute('href');
            if (href && href.startsWith('#')) {
                const stepName = href.substring(1);
                const stepMap = { home: 0, setup: 1, report: 2, voice: 3, reports: 4 };
                if (stepName in stepMap) {
                    btn.onclick = (e) => {
                        e.preventDefault();
                        portal.goToStep(stepMap[stepName]);
                    };
                }
            }
        });

        // Emergency FAB button
        const emergencyFab = document.querySelector('.emergency-fab');
        if (emergencyFab) {
            emergencyFab.onclick = portal.quickEmergency;
        }

        // Feature cards navigation
        document.querySelectorAll('.feature-card').forEach(card => {
            if (card.onclick) return; // Skip if already has onclick
            
            if (card.querySelector('.feature-title')?.textContent?.includes('Submit Report')) {
                card.onclick = () => portal.goToStep(2);
            } else if (card.querySelector('.feature-title')?.textContent?.includes('Voice Reporting')) {
                card.onclick = () => portal.goToStep(3);
            } else if (card.querySelector('.feature-title')?.textContent?.includes('Track My Reports')) {
                card.onclick = () => portal.goToStep(4);
            }
        });
    }

    // --- CORE UI & NAVIGATION ---
    portal.goToStep = function(stepNumber) {
        if (typeof stepNumber !== 'number' || stepNumber < 0 || stepNumber > 4) return;
        
        document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
        const targetSection = document.getElementById(`step-${stepNumber}`);
        if (targetSection) {
            targetSection.classList.add('active');
            document.querySelector('.main-content')?.scrollTo(0, 0);
        }
        
        updateStepIndicator(stepNumber);
        portal.currentStep = stepNumber;
        
        switch(stepNumber) {
            case 1: portal.checkExistingPermissions(); break;
            case 2: portal.initializeReportForm(); break;
            case 3: portal.initializeVoiceReporter(); break;
            case 4: portal.loadUserReports(); break;
        }
    };

    function updateStepIndicator(activeStep) {
        const steps = document.querySelectorAll('.step');
        const navBtns = document.querySelectorAll('.nav-btn');
        const progressFill = document.getElementById('progressFill');
        
        steps.forEach((step, index) => {
            step.classList.remove('active', 'completed');
            if (index < activeStep) step.classList.add('completed');
            if (index === activeStep) step.classList.add('active');
        });
        
        navBtns.forEach(btn => {
            const stepName = btn.getAttribute('href')?.substring(1);
            const stepMap = { home: 0, setup: 1, report: 2, voice: 3, reports: 4 };
            btn.classList.toggle('active', stepMap[stepName] === activeStep);
        });

        if (progressFill) {
            const progress = activeStep > 0 ? (activeStep / (steps.length - 1)) * 100 : 10;
            progressFill.style.width = `${progress}%`;
        }
    }

    // --- DYNAMIC FEATURES & UI UPDATES ---
    portal.checkLocalAlerts = async function() {
        const alertBanner = document.getElementById('localAlerts');
        const alertText = document.getElementById('alertText');
        if (alertBanner && alertText && Math.random() > 0.5) {
            alertText.textContent = 'Severe thunderstorm warning in effect for your area until 8 PM.';
            alertBanner.style.display = 'flex';
        }
    };

    portal.dismissAlert = function() {
        document.getElementById('localAlerts')?.style.display = 'none';
    };
    
    portal.showRiskPrediction = function() {
        portal.showNotification('🔮 AI-powered hazard prediction feature is coming soon!', 'info');
    };

    portal.showAIConfidence = function(confidence) {
        const meter = document.getElementById('aiConfidence');
        const scoreEl = document.getElementById('confidenceScore');
        const fillEl = document.getElementById('confidenceFill');
        if (!meter || !scoreEl || !fillEl) return;

        meter.style.display = 'block';
        scoreEl.textContent = `${confidence}%`;
        fillEl.style.width = `${confidence}%`;
        if (confidence >= 80) fillEl.style.background = '#16a34a';
        else if (confidence >= 60) fillEl.style.background = '#f59e0b';
        else fillEl.style.background = '#dc2626';
    };

    portal.updateAIStatus = function(status) {
        const statusBar = document.getElementById('aiStatusBar');
        const statusDot = document.getElementById('aiStatusDot');
        const statusText = document.getElementById('aiStatusText');
        
        if (!statusBar || !statusDot || !statusText) return;

        statusDot.className = `ai-status-dot ${status}`;
        switch(status) {
            case 'ready':
                statusText.textContent = '🧠 AI models ready for analysis';
                break;
            case 'loading':
                statusText.textContent = '🧠 Processing emergency data...';
                break;
            case 'error':
                statusText.textContent = '🧠 AI processing error - using fallback';
                break;
        }
    };

    // --- PERMISSIONS HANDLING ---
    portal.requestLocation = async function(button) {
        if (!navigator.geolocation) return portal.showNotification('Geolocation is not supported.', 'error');
        button.textContent = 'Requesting...';
        button.disabled = true;
        try {
            const pos = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, { 
                    timeout: 10000,
                    enableHighAccuracy: true,
                    maximumAge: 60000
                });
            });
            portal.userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
            portal.permissionsGranted.location = true;
            const statusEl = document.getElementById('locationStatus');
            if (statusEl) statusEl.textContent = `✅ Location: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`;
            portal.showPermissionSuccess(button, 'Location');
        } catch (err) {
            console.error('Location error:', err);
            portal.showPermissionError(button, 'Location');
            const statusEl = document.getElementById('locationStatus');
            if (statusEl) statusEl.textContent = '❌ Location access denied or unavailable';
        }
    };
    
    portal.requestMicrophone = async (btn) => {
        try {
            btn.textContent = 'Requesting...';
            btn.disabled = true;
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            portal.permissionsGranted.microphone = true;
            portal.showPermissionSuccess(btn, 'Microphone');
        } catch (err) {
            console.error('Microphone error:', err);
            portal.showPermissionError(btn, 'Microphone');
        }
    };
    
    portal.requestCamera = async (btn) => {
        try {
            btn.textContent = 'Requesting...';
            btn.disabled = true;
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());
            portal.permissionsGranted.camera = true;
            portal.showPermissionSuccess(btn, 'Camera');
        } catch (err) {
            console.error('Camera error:', err);
            portal.showPermissionError(btn, 'Camera');
        }
    };

    portal.requestNotifications = async (btn) => {
        if (!('Notification' in window)) return portal.showPermissionError(btn, 'Notifications');
        try {
            btn.textContent = 'Requesting...';
            btn.disabled = true;
            const permission = await Notification.requestPermission();
            if (permission === 'granted') {
                portal.permissionsGranted.notifications = true;
                portal.showPermissionSuccess(btn, 'Notifications');
                new Notification('Emergency Assistant', { 
                    body: 'Alerts enabled!',
                    icon: '/static/icons/notification-icon.png'
                });
            } else {
                portal.showPermissionError(btn, 'Notifications');
            }
        } catch (err) {
            console.error('Notification error:', err);
            portal.showPermissionError(btn, 'Notifications');
        }
    };

    portal.showPermissionSuccess = (button, type) => {
        if (!button) return;
        button.textContent = `✅ ${type} Enabled`;
        button.style.background = '#16a34a';
        button.style.color = 'white';
        button.disabled = true;
    };
    
    portal.showPermissionError = (button, type) => {
        if (!button) return;
        button.textContent = `❌ ${type} Denied`;
        button.style.background = '#dc2626';
        button.style.color = 'white';
        setTimeout(() => {
            button.textContent = `Enable ${type}`;
            button.style.background = '';
            button.style.color = '';
            button.disabled = false;
        }, 3000);
    };

    portal.checkExistingPermissions = async function() {
        // Check geolocation
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(pos => {
                portal.userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                portal.permissionsGranted.location = true;
                const statusEl = document.getElementById('locationStatus');
                if (statusEl) statusEl.textContent = `✅ Location: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`;
                const btn = document.querySelector('button[onclick*="requestLocation"]');
                if (btn) portal.showPermissionSuccess(btn, 'Location');
            }, () => {}, { timeout: 1000 });
        }

        // Check notification permission
        if ('Notification' in window && Notification.permission === 'granted') {
            portal.permissionsGranted.notifications = true;
            const btn = document.querySelector('button[onclick*="requestNotifications"]');
            if (btn) portal.showPermissionSuccess(btn, 'Notifications');
        }
    };

    // --- REPORT FORM LOGIC ---
    portal.initializeReportForm = function() {
        const selectedCard = document.querySelector('.method-card.selected');
        if (selectedCard) {
            const method = selectedCard.getAttribute('data-method') || 'text';
            portal.selectMethod(selectedCard, method);
        } else {
            const firstCard = document.querySelector('.method-card[data-method="text"]');
            if (firstCard) portal.selectMethod(firstCard, 'text');
        }
    };

    portal.selectMethod = function(element, method) {
        document.querySelectorAll('.method-card').forEach(c => c.classList.remove('selected'));
        element.classList.add('selected');
        
        const formContainer = document.getElementById('emergencyReportForm');
        if (!formContainer) return;

        let formHTML = `
            <div class="form-group">
                <label class="form-label" for="reportType">🚨 Emergency Type *</label>
                <select class="form-input" id="reportType" required>
                    <option value="">Select type...</option>
                    <option value="fire">🔥 Fire</option>
                    <option value="medical">🏥 Medical Emergency</option>
                    <option value="accident">🚗 Vehicle Accident</option>
                    <option value="weather">🌪️ Severe Weather</option>
                    <option value="crime">🚔 Crime in Progress</option>
                    <option value="infrastructure">🔧 Infrastructure Failure</option>
                    <option value="other">❓ Other Emergency</option>
                </select>
            </div>
            <div class="form-group">
                <label class="form-label" for="description">📝 Description *</label>
                <textarea class="form-input form-textarea" id="description" placeholder="Describe what is happening in detail..." required rows="4"></textarea>
            </div>
            <div class="form-group">
                <label class="form-label" for="location">📍 Location *</label>
                <input type="text" class="form-input" id="location" placeholder="Street address or landmark" required>
                <button type="button" class="btn btn-secondary" onclick="portal.useCurrentLocation()" style="margin-top: 0.5rem;">📍 Use My Location</button>
            </div>
            <div class="form-group">
                <label class="form-label" for="priorityLevel">⚠️ Priority Level</label>
                <select class="form-input" id="priorityLevel">
                    <option value="low">🟢 Low - Non-urgent situation</option>
                    <option value="medium" selected>🟡 Medium - Needs attention</option>
                    <option value="high">🟠 High - Urgent response needed</option>
                    <option value="critical">🔴 Critical - Life threatening</option>
                </select>
            </div>
        `;

        if (method === 'photo') {
            formHTML += `
                <div class="form-group">
                    <label class="form-label" for="evidenceFile">📷 Upload Photo Evidence *</label>
                    <input type="file" class="form-input" id="evidenceFile" accept="image/*" required>
                    <div id="imagePreview" style="margin-top: 1rem; display: none;">
                        <img id="previewImg" style="max-width: 200px; max-height: 200px; border-radius: 8px;">
                        <button type="button" class="btn btn-secondary" onclick="portal.removeImage()" style="margin-left: 1rem;">🗑️ Remove</button>
                    </div>
                </div>
            `;
        } else if (method === 'text' || method === 'location') {
            formHTML += `
                <div class="form-group">
                    <label class="form-label">📷 Photo Evidence (Optional)</label>
                    <input type="file" class="form-input" id="evidenceFile" accept="image/*">
                    <div id="imagePreview" style="margin-top: 1rem; display: none;">
                        <img id="previewImg" style="max-width: 200px; max-height: 200px; border-radius: 8px;">
                        <button type="button" class="btn btn-secondary" onclick="portal.removeImage()" style="margin-left: 1rem;">🗑️ Remove</button>
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">🎵 Audio Note (Optional)</label>
                    <div class="audio-recorder">
                        <button type="button" class="btn btn-secondary" id="audioRecordBtn" onclick="portal.toggleAudioRecording()">🎤 Record Audio Note</button>
                        <div id="audioPlayback" class="audio-playback" style="display: none; margin-top: 1rem;">
                            <audio controls id="recordedAudio" style="width: 100%;"></audio>
                            <button type="button" class="btn btn-secondary" onclick="portal.removeAudio()" style="margin-top: 0.5rem;">🗑️ Remove Audio</button>
                        </div>
                    </div>
                </div>
            `;
        }

        formContainer.innerHTML = formHTML;

        // Setup file upload preview
        const fileInput = document.getElementById('evidenceFile');
        if (fileInput) {
            fileInput.onchange = portal.handleFileUpload;
        }

        // Setup AI analysis simulation
        const description = document.getElementById('description');
        if (description) {
            description.oninput = portal.simulateAIAnalysis;
        }

        portal.showAIConfidence(Math.floor(Math.random() * 20) + 75);
    };

    portal.handleFileUpload = function(event) {
        const file = event.target.files[0];
        if (!file) return;

        const preview = document.getElementById('imagePreview');
        const img = document.getElementById('previewImg');
        
        if (preview && img) {
            const reader = new FileReader();
            reader.onload = function(e) {
                img.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        portal.simulateImageAnalysis();
    };

    portal.removeImage = function() {
        const fileInput = document.getElementById('evidenceFile');
        const preview = document.getElementById('imagePreview');
        
        if (fileInput) fileInput.value = '';
        if (preview) preview.style.display = 'none';
    };

    portal.useCurrentLocation = function() {
        if (portal.userLocation) {
            const locationInput = document.getElementById('location');
            if (locationInput) {
                locationInput.value = `${portal.userLocation.lat.toFixed(6)}, ${portal.userLocation.lng.toFixed(6)}`;
                portal.showNotification('📍 Current location added', 'success');
            }
        } else {
            portal.showNotification('📍 Location not available. Please enable location permissions.', 'warning');
        }
    };

    portal.simulateAIAnalysis = function() {
        const description = document.getElementById('description').value;
        if (description.length < 10) return;

        portal.updateAIStatus('loading');
        
        setTimeout(() => {
            const urgencyKeywords = ['fire', 'bleeding', 'unconscious', 'explosion', 'trapped', 'dying', 'critical'];
            const medicalKeywords = ['injury', 'hurt', 'pain', 'bleeding', 'broken', 'unconscious'];
            const fireKeywords = ['fire', 'smoke', 'burning', 'flames', 'explosion'];

            let confidence = 60 + Math.random() * 30;
            let detectedType = 'other';
            
            if (urgencyKeywords.some(word => description.toLowerCase().includes(word))) {
                confidence += 15;
            }
            
            if (medicalKeywords.some(word => description.toLowerCase().includes(word))) {
                detectedType = 'medical';
                confidence += 10;
            } else if (fireKeywords.some(word => description.toLowerCase().includes(word))) {
                detectedType = 'fire';
                confidence += 10;
            }

            // Auto-update form based on AI analysis
            const typeSelect = document.getElementById('reportType');
            const prioritySelect = document.getElementById('priorityLevel');
            
            if (typeSelect && typeSelect.value === '') {
                typeSelect.value = detectedType;
            }
            
            if (urgencyKeywords.some(word => description.toLowerCase().includes(word)) && prioritySelect) {
                prioritySelect.value = 'high';
            }

            portal.showAIConfidence(Math.floor(confidence));
            portal.updateAIStatus('ready');
            
            portal.showNotification(`🤖 AI detected: ${detectedType} emergency (${Math.floor(confidence)}% confidence)`, 'info');
        }, 1500);
    };

    portal.simulateImageAnalysis = function() {
        portal.updateAIStatus('loading');
        portal.showNotification('🖼️ Analyzing image for hazards...', 'info');
        
        setTimeout(() => {
            const hazards = ['Fire visible', 'Structural damage', 'Smoke detected'];
            const randomHazard = hazards[Math.floor(Math.random() * hazards.length)];
            portal.showNotification(`🤖 Image analysis: ${randomHazard}`, 'warning');
            portal.updateAIStatus('ready');
        }, 2000);
    };

    portal.submitReport = async function() {
        const form = document.getElementById('emergencyReportForm');
        const type = document.getElementById('reportType')?.value;
        const description = document.getElementById('description')?.value;
        const location = document.getElementById('location')?.value;
        const priority = document.getElementById('priorityLevel')?.value || 'medium';

        if (!type || !description || !location) {
            portal.showNotification('❌ Please fill in all required fields', 'error');
            return;
        }

        portal.updateAIStatus('loading');
        
        try {
            // Simulate API call
            portal.showNotification('📤 Submitting emergency report...', 'info');
            
            // Create report data
            const reportData = {
                id: Date.now(),
                type,
                description,
                location,
                priority,
                timestamp: new Date().toISOString(),
                status: 'submitted',
                method: 'text'
            };

            // Save to local storage for tracking
            const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
            savedReports.unshift(reportData);
            localStorage.setItem('userReports', JSON.stringify(savedReports.slice(0, 50))); // Keep last 50

            setTimeout(() => {
                portal.showNotification('✅ Emergency report submitted successfully!', 'success');
                portal.updateAIStatus('ready');
                
                // Show submission confirmation
                const confirmation = `
                    <div style="background: #f0f9ff; border: 2px solid #3b82f6; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #1e40af;">📝 Report Submitted</h4>
                        <p><strong>Report ID:</strong> ER-${reportData.id}</p>
                        <p><strong>Type:</strong> ${type}</p>
                        <p><strong>Priority:</strong> ${priority}</p>
                        <p><strong>Status:</strong> Processing</p>
                        <p><strong>Estimated Response:</strong> ${priority === 'critical' ? '5-10 minutes' : priority === 'high' ? '15-30 minutes' : '30-60 minutes'}</p>
                    </div>
                `;
                
                const formContainer = document.getElementById('emergencyReportForm');
                if (formContainer) {
                    formContainer.innerHTML = confirmation;
                }
                
                setTimeout(() => portal.goToStep(4), 2000);
            }, 1500);
            
        } catch (error) {
            console.error('Submit error:', error);
            portal.showNotification('❌ Failed to submit report. Please try again.', 'error');
            portal.updateAIStatus('error');
        }
    };

    // --- VOICE REPORTING ---
    portal.setupSpeechRecognition = function() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            portal.recognition = new SpeechRecognition();
            portal.recognition.continuous = true;
            portal.recognition.interimResults = true;
            portal.recognition.lang = 'en-US';
            
            portal.recognition.onresult = function(event) {
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
                    transcriptBox.textContent = finalTranscript + interimTranscript;
                    portal.voiceTranscript = finalTranscript;
                }
                
                if (finalTranscript) {
                    portal.analyzeVoiceContent(finalTranscript);
                }
            };
            
            portal.recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                portal.showNotification('🎤 Voice recognition error. Please try again.', 'error');
                portal.stopVoiceRecording();
            };
            
            portal.recognition.onend = function() {
                if (portal.isVoiceRecording) {
                    // Restart recognition if still recording
                    portal.recognition.start();
                }
            };
        }
    };

    portal.initializeVoiceReporter = function() {
        const transcriptBox = document.getElementById('transcriptBox');
        const voiceStatus = document.getElementById('voiceStatus');
        const submitBtn = document.getElementById('submitVoiceReport');
        
        if (transcriptBox) transcriptBox.textContent = 'Your voice transcript will appear here...';
        if (voiceStatus) voiceStatus.textContent = 'Tap the microphone to start recording';
        if (submitBtn) submitBtn.disabled = true;
        
        portal.voiceTranscript = '';
        portal.currentAnalysis = {};
    };

    portal.toggleRecording = function() {
        if (portal.isVoiceRecording) {
            portal.stopVoiceRecording();
        } else {
            portal.startVoiceRecording();
        }
    };

    portal.startVoiceRecording = async function() {
        if (!portal.permissionsGranted.microphone) {
            portal.showNotification('🎤 Please enable microphone access first', 'warning');
            return;
        }

        try {
            const recordButton = document.getElementById('recordButton');
            const voiceStatus = document.getElementById('voiceStatus');
            
            portal.isVoiceRecording = true;
            
            if (recordButton) {
                recordButton.textContent = '⏹️';
                recordButton.style.background = '#dc2626';
            }
            
            if (voiceStatus) voiceStatus.textContent = '🎤 Recording... Speak clearly about the emergency';
            
            if (portal.recognition) {
                portal.recognition.start();
            }
            
            portal.showNotification('🎤 Voice recording started', 'success');
            
        } catch (error) {
            console.error('Voice recording error:', error);
            portal.showNotification('❌ Could not start voice recording', 'error');
            portal.stopVoiceRecording();
        }
    };

    portal.stopVoiceRecording = function() {
        portal.isVoiceRecording = false;
        
        const recordButton = document.getElementById('recordButton');
        const voiceStatus = document.getElementById('voiceStatus');
        
        if (recordButton) {
            recordButton.textContent = '🎤';
            recordButton.style.background = '';
        }
        
        if (voiceStatus) voiceStatus.textContent = 'Recording stopped. Processing analysis...';
        
        if (portal.recognition) {
            portal.recognition.stop();
        }
        
        setTimeout(() => {
            const submitBtn = document.getElementById('submitVoiceReport');
            if (submitBtn && portal.voiceTranscript) {
                submitBtn.disabled = false;
            }
            
            if (voiceStatus) {
                voiceStatus.textContent = portal.voiceTranscript ? 'Recording complete. Review and submit.' : 'Tap to start new recording';
            }
        }, 1000);
    };

    portal.analyzeVoiceContent = function(transcript) {
        // Simulate real-time voice analysis
        const urgencyKeywords = ['help', 'emergency', 'fire', 'accident', 'critical', 'urgent', 'dying', 'trapped'];
        const medicalKeywords = ['hurt', 'injured', 'bleeding', 'unconscious', 'pain', 'broken'];
        const emotionKeywords = {
            panic: ['panic', 'scared', 'terrified', 'help me'],
            calm: ['stable', 'okay', 'fine', 'controlled'],
            urgent: ['hurry', 'fast', 'quickly', 'immediately', 'now']
        };
        
        // Analyze urgency
        const urgencyMatches = urgencyKeywords.filter(word => transcript.toLowerCase().includes(word));
        let urgencyLevel = urgencyMatches.length >= 3 ? 'critical' : urgencyMatches.length >= 2 ? 'high' : urgencyMatches.length >= 1 ? 'medium' : 'low';
        
        // Analyze emotion
        let emotionLevel = 'calm';
        for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
            if (keywords.some(word => transcript.toLowerCase().includes(word))) {
                emotionLevel = emotion;
                break;
            }
        }
        
        // Detect emergency type
        let emergencyType = 'general';
        if (medicalKeywords.some(word => transcript.toLowerCase().includes(word))) {
            emergencyType = 'medical';
        } else if (['fire', 'burning', 'smoke'].some(word => transcript.toLowerCase().includes(word))) {
            emergencyType = 'fire';
        } else if (['accident', 'crash', 'collision'].some(word => transcript.toLowerCase().includes(word))) {
            emergencyType = 'accident';
        }
        
        // Extract location if mentioned
        let detectedLocation = '';
        const locationPatterns = [
            /at (.+?)(?:\s|$)/gi,
            /near (.+?)(?:\s|$)/gi,
            /on (.+?)(?:\s|$)/gi
        ];
        
        for (const pattern of locationPatterns) {
            const match = transcript.match(pattern);
            if (match && match[1] && match[1].length > 3) {
                detectedLocation = match[1].trim();
                break;
            }
        }
        
        // Generate AI recommendation
        let recommendation = 'Continue monitoring situation';
        if (urgencyLevel === 'critical') {
            recommendation = 'Dispatch emergency responders immediately';
        } else if (urgencyLevel === 'high') {
            recommendation = 'Prioritize for rapid response';
        } else if (emergencyType === 'medical') {
            recommendation = 'Send medical assistance';
        } else if (emergencyType === 'fire') {
            recommendation = 'Alert fire department';
        }
        
        // Update UI with analysis
        portal.currentAnalysis = {
            urgency: urgencyLevel,
            emotion: emotionLevel,
            type: emergencyType,
            location: detectedLocation,
            recommendation: recommendation
        };
        
        portal.updateVoiceAnalysisUI();
    };

    portal.updateVoiceAnalysisUI = function() {
        const analysisSection = document.getElementById('voiceAnalysis');
        if (analysisSection && portal.currentAnalysis.urgency) {
            analysisSection.style.display = 'block';
            
            const urgencyEl = document.getElementById('urgencyLevel');
            const emotionEl = document.getElementById('emotionLevel');
            const locationEl = document.getElementById('detectedLocation');
            const recommendationEl = document.getElementById('aiRecommendation');
            
            if (urgencyEl) {
                urgencyEl.textContent = portal.currentAnalysis.urgency.toUpperCase();
                urgencyEl.style.color = {
                    'critical': '#dc2626',
                    'high': '#f59e0b',
                    'medium': '#3b82f6',
                    'low': '#16a34a'
                }[portal.currentAnalysis.urgency] || '#6b7280';
            }
            
            if (emotionEl) emotionEl.textContent = portal.currentAnalysis.emotion;
            if (locationEl) locationEl.textContent = portal.currentAnalysis.location || 'Not detected';
            if (recommendationEl) recommendationEl.textContent = portal.currentAnalysis.recommendation;
        }
    };

    portal.submitVoiceReport = async function() {
        if (!portal.voiceTranscript) {
            portal.showNotification('❌ No voice transcript available', 'error');
            return;
        }
        
        try {
            portal.showNotification('📤 Submitting voice report...', 'info');
            
            // Create voice report data
            const reportData = {
                id: Date.now(),
                type: 'voice_emergency',
                transcript: portal.voiceTranscript,
                urgency: portal.currentAnalysis.urgency || 'medium',
                emotion: portal.currentAnalysis.emotion || 'neutral',
                location: portal.currentAnalysis.location || 'Not specified',
                recommendation: portal.currentAnalysis.recommendation || '',
                timestamp: new Date().toISOString(),
                status: 'submitted',
                method: 'voice'
            };
            
            // Save to local storage
            const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
            savedReports.unshift(reportData);
            localStorage.setItem('userReports', JSON.stringify(savedReports.slice(0, 50)));
            
            setTimeout(() => {
                portal.showNotification('✅ Voice report submitted successfully!', 'success');
                
                // Show confirmation
                const confirmation = `
                    <div style="background: #f0f9ff; border: 2px solid #3b82f6; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #1e40af;">🎤 Voice Report Submitted</h4>
                        <p><strong>Report ID:</strong> VR-${reportData.id}</p>
                        <p><strong>Urgency:</strong> ${reportData.urgency}</p>
                        <p><strong>Transcript:</strong> "${reportData.transcript.substring(0, 100)}..."</p>
                        <p><strong>Status:</strong> Processing</p>
                    </div>
                `;
                
                const analysisSection = document.getElementById('voiceAnalysis');
                if (analysisSection) {
                    analysisSection.innerHTML = confirmation;
                }
                
                setTimeout(() => portal.goToStep(4), 2000);
            }, 1500);
            
        } catch (error) {
            console.error('Voice submit error:', error);
            portal.showNotification('❌ Failed to submit voice report', 'error');
        }
    };

    // --- AUDIO RECORDING FOR FORM NOTES ---
    portal.toggleAudioRecording = async function() {
        const recordBtn = document.getElementById('audioRecordBtn');
        if (portal.isAudioNoteRecording) {
            portal.mediaRecorder?.stop();
        } else {
            try {
                recordBtn.textContent = 'Requesting...';
                recordBtn.disabled = true;
                
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                portal.mediaRecorder = new MediaRecorder(stream);
                portal.audioChunks = [];
                
                portal.mediaRecorder.ondataavailable = e => portal.audioChunks.push(e.data);
                portal.mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(portal.audioChunks, { type: 'audio/wav' });
                    const audioURL = URL.createObjectURL(audioBlob);
                    document.getElementById('recordedAudio').src = audioURL;
                    document.getElementById('audioPlayback').style.display = 'block';
                    stream.getTracks().forEach(track => track.stop());
                    portal.isAudioNoteRecording = false;
                    updateRecordButtonUI(false);
                };
                
                portal.mediaRecorder.start();
                portal.isAudioNoteRecording = true;
                updateRecordButtonUI(true);
                
            } catch (err) {
                console.error('Audio recording error:', err);
                portal.showNotification('❌ Could not access microphone for audio note', 'error');
                recordBtn.textContent = '🎤 Record Audio Note';
                recordBtn.disabled = false;
            }
        }
    };
    
    function updateRecordButtonUI(isRecording) {
        const btn = document.getElementById('audioRecordBtn');
        if (btn) {
            btn.textContent = isRecording ? '⏹️ Stop Recording' : '🎤 Record Audio Note';
            btn.style.backgroundColor = isRecording ? '#dc2626' : '';
            btn.style.color = isRecording ? 'white' : '';
            btn.disabled = false;
        }
    }

    portal.removeAudio = function() {
        portal.audioChunks = [];
        const audio = document.getElementById('recordedAudio');
        const playback = document.getElementById('audioPlayback');
        
        if (audio) audio.src = '';
        if (playback) playback.style.display = 'none';
    };

    // --- REPORTS TRACKING ---
    portal.loadUserReports = function() {
        const reportsGrid = document.getElementById('reportsGrid');
        if (!reportsGrid) return;
        
        const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
        
        if (savedReports.length === 0) {
            reportsGrid.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: #6b7280;">
                    <h3>📋 No Reports Yet</h3>
                    <p>You haven't submitted any emergency reports yet.</p>
                    <button class="btn btn-primary" onclick="portal.goToStep(2)" style="margin-top: 1rem;">
                        📱 Submit Your First Report
                    </button>
                </div>
            `;
            return;
        }
        
        const reportsHTML = savedReports.map((report, index) => {
            const timeAgo = portal.getTimeAgo(new Date(report.timestamp));
            const statusColor = {
                'submitted': '#3b82f6',
                'processing': '#f59e0b',
                'responded': '#16a34a',
                'closed': '#6b7280'
            }[report.status] || '#6b7280';
            
            const priorityIcon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }[report.urgency || report.priority] || '🟡';
            
            return `
                <div class="report-card" style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: white;">
                    <div style="display: flex; justify-content: between; align-items: start; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #1f2937;">
                            ${report.method === 'voice' ? '🎤' : '📝'} Report #${index + 1}
                        </h4>
                        <span style="background: ${statusColor}; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                            ${report.status}
                        </span>
                    </div>
                    <p style="margin: 0.5rem 0; color: #6b7280; font-size: 0.9rem;">
                        ${priorityIcon} ${(report.urgency || report.priority || 'medium').toUpperCase()} Priority
                    </p>
                    <p style="margin: 0.5rem 0; color: #374151;">
                        ${report.method === 'voice' ? report.transcript?.substring(0, 100) + '...' : (report.description?.substring(0, 100) + '...' || 'No description')}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; font-size: 0.8rem; color: #6b7280;">
                        <span>📍 ${report.location || 'No location'}</span>
                        <span>⏰ ${timeAgo}</span>
                    </div>
                    <div style="margin-top: 1rem;">
                        <button class="btn btn-secondary" onclick="portal.viewReportDetails(${index})" style="margin-right: 0.5rem;">
                            👁️ View Details
                        </button>
                        <button class="btn btn-secondary" onclick="portal.trackReport('${report.id}')">
                            📊 Track Status
                        </button>
                    </div>
                </div>
            `;
        }).join('');
        
        reportsGrid.innerHTML = reportsHTML;
        
        // Add filter controls
        const filtersContainer = document.querySelector('.reports-filters');
        if (filtersContainer) {
            filtersContainer.innerHTML = `
                <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem;">
                    <label>Filter by:</label>
                    <select id="statusFilter" onchange="portal.filterReports()" style="padding: 0.5rem; border: 1px solid #d1d5db; border-radius: 4px;">
                        <option value="">All Status</option>
                        <option value="submitted">Submitted</option>
                        <option value="processing">Processing</option>
                        <option value="responded">Responded</option>
                        <option value="closed">Closed</option>
                    </select>
                    <select id="methodFilter" onchange="portal.filterReports()" style="padding: 0.5rem; border: 1px solid #d1d5db; border-radius: 4px;">
                        <option value="">All Methods</option>
                        <option value="text">Text Reports</option>
                        <option value="voice">Voice Reports</option>
                    </select>
                    <button class="btn btn-secondary" onclick="portal.exportReports()">
                        📁 Export Reports
                    </button>
                </div>
            `;
        }
    };

    portal.getTimeAgo = function(date) {
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} minutes ago`;
        if (diffHours < 24) return `${diffHours} hours ago`;
        return `${diffDays} days ago`;
    };

    portal.viewReportDetails = function(index) {
        const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
        const report = savedReports[index];
        if (!report) return;
        
        const details = `
            📋 REPORT DETAILS
            
            Report ID: ${report.method === 'voice' ? 'VR' : 'ER'}-${report.id}
            Type: ${report.type || 'Emergency'}
            Method: ${report.method === 'voice' ? 'Voice Recording' : 'Text Report'}
            Priority: ${(report.urgency || report.priority || 'medium').toUpperCase()}
            Status: ${report.status}
            
            ${report.method === 'voice' ? 'Transcript:' : 'Description:'}
            ${report.transcript || report.description || 'No details available'}
            
            Location: ${report.location || 'Not specified'}
            Submitted: ${new Date(report.timestamp).toLocaleString()}
            
            ${report.recommendation ? 'AI Recommendation: ' + report.recommendation : ''}
        `;
        
        alert(details);
    };

    portal.trackReport = function(reportId) {
        // Simulate report tracking
        const statuses = ['submitted', 'processing', 'responded'];
        const currentStatus = statuses[Math.floor(Math.random() * statuses.length)];
        
        portal.showNotification(`📊 Report ${reportId} status: ${currentStatus.toUpperCase()}`, 'info');
    };

    portal.filterReports = function() {
        const statusFilter = document.getElementById('statusFilter')?.value;
        const methodFilter = document.getElementById('methodFilter')?.value;
        
        // Simple implementation - reload reports with filters
        // In a real app, this would filter the display
        console.log('Filtering reports:', { status: statusFilter, method: methodFilter });
        portal.loadUserReports();
    };

    portal.exportReports = function() {
        const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
        if (savedReports.length === 0) {
            portal.showNotification('📁 No reports to export', 'warning');
            return;
        }
        
        const dataStr = JSON.stringify(savedReports, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `emergency-reports-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        portal.showNotification('📁 Reports exported successfully', 'success');
    };

    // --- OFFLINE HANDLING ---
    portal.setupOfflineHandling = function() {
        // Register service worker for offline support
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js')
                .then(registration => console.log('✅ Service Worker registered'))
                .catch(error => console.log('❌ Service Worker registration failed:', error));
        }
        
        // Handle online/offline events
        window.addEventListener('online', () => {
            portal.showNotification('🌐 Back online! Syncing data...', 'success');
            portal.syncOfflineData();
        });
        
        window.addEventListener('offline', () => {
            portal.showNotification('📴 Offline mode active. Reports will be saved locally.', 'warning');
        });
    };

    portal.syncOfflineData = function() {
        // This would sync any offline data with the server
        console.log('🔄 Syncing offline data...');
    };

    // --- UTILITY FUNCTIONS ---
    portal.showNotification = function(message, type = 'info') {
        // Remove existing notifications
        document.querySelectorAll('.notification').forEach(n => n.remove());
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 10000;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            animation: slideIn 0.3s ease-out;
        `;
        
        const colors = {
            success: '#16a34a',
            error: '#dc2626',
            warning: '#f59e0b',
            info: '#3b82f6'
        };
        
        notification.style.backgroundColor = colors[type] || colors.info;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => notification.remove(), 300);
        }, 4000);
    };

    portal.quickEmergency = function() {
        if (confirm('🚨 QUICK EMERGENCY REPORT\n\nThis will take you directly to the emergency report form with CRITICAL priority.\n\nProceed?')) {
            portal.goToStep(2);
            setTimeout(() => {
                const prioritySelect = document.getElementById('priorityLevel');
                if (prioritySelect) {
                    prioritySelect.value = 'critical';
                    portal.showNotification('⚠️ Priority set to CRITICAL', 'warning');
                }
            }, 500);
        }
    };

    // Add CSS animations
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
        .report-card {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .report-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
    `;
    document.head.appendChild(style);

    // --- Make portal object globally accessible ---
    window.portal = portal;
    
    // --- Initialize the application ---
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeApp);
    } else {
        initializeApp();
    }

})();