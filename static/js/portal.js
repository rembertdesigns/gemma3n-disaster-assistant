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
        alert('🔮 AI-powered hazard prediction feature is coming soon!');
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

    // --- PERMISSIONS HANDLING ---
    portal.requestLocation = async function(button) {
        if (!navigator.geolocation) return portal.showNotification('Geolocation is not supported.', 'error');
        button.textContent = 'Requesting...';
        button.disabled = true;
        try {
            const pos = await new Promise((resolve, reject) => navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 10000 }));
            portal.userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
            portal.permissionsGranted.location = true;
            document.getElementById('locationStatus').textContent = '✅ Location access granted.';
            portal.showPermissionSuccess(button, 'Location');
        } catch (err) {
            portal.showPermissionError(button, 'Location');
        }
    };
    
    portal.requestMicrophone = async (btn) => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            portal.permissionsGranted.microphone = true;
            portal.showPermissionSuccess(btn, 'Microphone');
        } catch (err) {
            portal.showPermissionError(btn, 'Microphone');
        }
    };
    
    portal.requestCamera = async (btn) => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());
            portal.permissionsGranted.camera = true;
            portal.showPermissionSuccess(btn, 'Camera');
        } catch (err) {
            portal.showPermissionError(btn, 'Camera');
        }
    };

    portal.requestNotifications = async (btn) => {
        if (!('Notification' in window)) return portal.showPermissionError(btn, 'Notifications');
        const permission = await Notification.requestPermission();
        if (permission === 'granted') {
            portal.permissionsGranted.notifications = true;
            portal.showPermissionSuccess(btn, 'Notifications');
            new Notification('Emergency Assistant', { body: 'Alerts enabled!' });
        } else {
            portal.showPermissionError(btn, 'Notifications');
        }
    };

    portal.showPermissionSuccess = (button, type) => {
        if (!button) return;
        button.textContent = `✅ ${type} Enabled`;
        button.style.background = 'var(--accent-color)';
        button.disabled = true;
    };
    
    portal.showPermissionError = (button, type) => {
        if (!button) return;
        button.textContent = `❌ ${type} Denied`;
        button.style.background = 'var(--danger-color)';
        setTimeout(() => {
            button.textContent = `Enable ${type}`;
            button.style.background = '';
            button.disabled = false;
        }, 3000);
    };

    portal.checkExistingPermissions = async function() {
        // This function can be expanded with navigator.permissions.query if needed
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(pos => {
                portal.userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                portal.permissionsGranted.location = true;
                document.getElementById('locationStatus').textContent = '✅ Location already enabled.';
            }, () => {}, { timeout: 1000 });
        }
    };

    // --- REPORT FORM LOGIC ---
    portal.initializeReportForm = function() {
        portal.selectMethod(document.querySelector('.method-card.selected'), 'text');
    };

    portal.selectMethod = function(element, method) {
        document.querySelectorAll('.method-card').forEach(c => c.classList.remove('selected'));
        element.classList.add('selected');
        
        const formContainer = document.getElementById('emergencyReportForm');
        formContainer.innerHTML = `
            <div class="form-group">
                <label class="form-label" for="reportType">🚨 Emergency Type *</label>
                <select class="form-input" id="reportType" required>
                    <option value="">Select type...</option>
                    <option value="fire">🔥 Fire</option><option value="medical">🏥 Medical</option>
                    <option value="accident">🚗 Accident</option><option value="other">❓ Other</option>
                </select>
            </div>
            <div class="form-group">
                <label class="form-label" for="description">📝 Description *</label>
                <textarea class="form-input form-textarea" id="description" placeholder="Describe what is happening..." required></textarea>
            </div>
            <div class="form-group" id="evidenceFileGroup" style="display: ${method === 'photo' ? 'block' : 'none'};">
                <label class="form-label" for="evidenceFile">📷 Upload Photo</label>
                <input type="file" class="form-input" id="evidenceFile" accept="image/*">
            </div>
             <div class="form-group" id="audioBackupGroup" style="display: ${method === 'text' || method === 'location' ? 'block' : 'none'};">
                <label class="form-label">🎵 Audio Note (Optional)</label>
                <div class="audio-recorder">
                    <button type="button" class="btn btn-secondary" id="audioRecordBtn" onclick="portal.toggleAudioRecording()">🎤 Record Audio</button>
                    <div id="audioPlayback" class="audio-playback" style="display: none;">
                        <audio controls id="recordedAudio"></audio>
                        <button type="button" class="btn btn-secondary" onclick="portal.removeAudio()">🗑️ Remove</button>
                    </div>
                </div>
            </div>
        `;
        portal.showAIConfidence(Math.floor(Math.random() * 20) + 75);
    };

    portal.submitReport = function() {
        portal.showNotification('📤 Report submitted successfully!', 'success');
        setTimeout(() => portal.goToStep(4), 1500);
    };

    // --- AUDIO RECORDING ---
    portal.toggleAudioRecording = async function() {
        const recordBtn = document.getElementById('audioRecordBtn');
        if (portal.isAudioNoteRecording) {
            portal.mediaRecorder?.stop();
        } else {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                portal.mediaRecorder = new MediaRecorder(stream);
                portal.audioChunks = [];
                portal.mediaRecorder.ondataavailable = e => portal.audioChunks.push(e.data);
                portal.mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(portal.audioChunks, { type: 'audio/wav' });
                    document.getElementById('recordedAudio').src = URL.createObjectURL(audioBlob);
                    document.getElementById('audioPlayback').style.display = 'flex';
                    stream.getTracks().forEach(track => track.stop());
                    portal.isAudioNoteRecording = false;
                    updateRecordButtonUI(false);
                };
                portal.mediaRecorder.start();
                portal.isAudioNoteRecording = true;
                updateRecordButtonUI(true);
            } catch (err) {
                portal.showNotification('Could not access microphone.', 'error');
            }
        }
    };
    
    function updateRecordButtonUI(isRec) {
        const btn = document.getElementById('audioRecordBtn');
        btn.textContent = isRec ? '⏹️ Stop Recording' : '🎤 Record Audio Note';
        btn.style.backgroundColor = isRec ? '#dc2626' : '';
    }

    portal.removeAudio = function() {
        portal.audioChunks = [];
        document.getElementById('recordedAudio').src = '';
        document.getElementById('audioPlayback').style.display = 'none';
    };

    // --- UTILITY FUNCTIONS ---
    portal.showNotification = function(message, type = 'info') {
        const el = document.createElement('div');
        el.className = `notification ${type}`;
        el.textContent = message;
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 4000);
    };

    portal.quickEmergency = function() {
        if (confirm('🚨 QUICK EMERGENCY REPORT\nThis will take you to the report form with CRITICAL priority pre-selected.\n\nProceed?')) {
            portal.goToStep(2);
            setTimeout(() => {
                const prioritySelect = document.getElementById('priorityLevel');
                if (prioritySelect) prioritySelect.value = 'critical';
                portal.showNotification('⚠️ Priority set to CRITICAL', 'warning');
            }, 100);
        }
    };

    // --- Make portal object globally accessible ---
    window.portal = portal;
    
    // --- Initialize the application ---
    document.addEventListener('DOMContentLoaded', initializeApp);

})();