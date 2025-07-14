// Enhanced Portal JavaScript - Setup First Approach
// Professional user flow prioritizing emergency profile setup

const portal = {
    currentStep: 0,
    userLocation: null,
    isVoiceRecording: false,
    voiceTranscript: '',
    permissionsGranted: { 
        location: false, 
        microphone: false, 
        camera: false, 
        notifications: false 
    },
    setupComplete: false,
    setupProgress: 0
};

// Enhanced step navigation with setup validation
portal.goToStep = function(stepNumber) {
    console.log(`Going to step ${stepNumber}`);
    
    // Check if user is trying to skip setup without emergency
    if (stepNumber > 0 && !portal.setupComplete && !portal.hasEmergencyOverride) {
        if (!confirm('⚠️ Setup not complete!\n\nYou can continue, but some features may not work properly. Are you sure?')) {
            return;
        }
    }
    
    // Hide all sections
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(s => s.classList.remove('active'));
    
    // Show target section
    const targetSection = document.getElementById(`step-${stepNumber}`);
    if (targetSection) {
        targetSection.classList.add('active');
    }
    
    // Update step indicator
    const steps = document.querySelectorAll('.step');
    steps.forEach((step, index) => {
        step.classList.remove('active', 'completed');
        if (index < stepNumber) step.classList.add('completed');
        if (index === stepNumber) step.classList.add('active');
    });
    
    // Update progress bar
    const progressFill = document.getElementById('progressFill');
    if (progressFill) {
        const progress = stepNumber > 0 ? (stepNumber / 4) * 100 : 20;
        progressFill.style.width = `${progress}%`;
    }
    
    portal.currentStep = stepNumber;
    
    // Initialize step-specific features
    if (stepNumber === 1) {
        portal.showWelcomeStats();
    } else if (stepNumber === 2) {
        portal.initializeReportForm();
    } else if (stepNumber === 4) {
        portal.loadUserReports();
    }
    
    // Save current step to localStorage
    localStorage.setItem('portalCurrentStep', stepNumber);
};

// Enhanced permission request functions with professional feedback
portal.requestLocationSetup = function(button) {
    if (!navigator.geolocation) {
        portal.showPermissionError(button, 'Geolocation not supported by this browser');
        return;
    }
    
    portal.setButtonRequesting(button, 'Requesting GPS access...');
    
    navigator.geolocation.getCurrentPosition(
        (position) => {
            portal.userLocation = { 
                lat: position.coords.latitude, 
                lng: position.coords.longitude,
                accuracy: position.coords.accuracy
            };
            portal.permissionsGranted.location = true;
            
            portal.setButtonGranted(button, '✅ GPS Access Granted');
            portal.updatePermissionStatus('locationSetupStatus', 
                `Location: ${position.coords.latitude.toFixed(4)}, ${position.coords.longitude.toFixed(4)} (±${Math.round(position.coords.accuracy)}m)`);
            portal.updatePermissionCard('locationSetupCard', 'granted');
            portal.updateSetupProgress();
            
            // Show success notification
            portal.showNotification('📍 Location access granted! Your emergency reports will include precise GPS coordinates.', 'success');
        },
        (error) => {
            let errorMessage = 'Location access denied';
            switch(error.code) {
                case error.PERMISSION_DENIED:
                    errorMessage = 'Location access denied by user';
                    break;
                case error.POSITION_UNAVAILABLE:
                    errorMessage = 'Location information unavailable';
                    break;
                case error.TIMEOUT:
                    errorMessage = 'Location request timed out';
                    break;
            }
            
            portal.setButtonDenied(button, '❌ GPS Access Denied');
            portal.updatePermissionStatus('locationSetupStatus', errorMessage);
            portal.showNotification('❌ ' + errorMessage + '. You can try again or continue without location access.', 'error');
            
            setTimeout(() => portal.resetButton(button, 'Enable GPS Access'), 4000);
        }
    );
};

portal.requestCameraSetup = function(button) {
    portal.setButtonRequesting(button, 'Requesting camera access...');
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            // Stop the stream immediately after getting permission
            stream.getTracks().forEach(track => track.stop());
            
            portal.permissionsGranted.camera = true;
            portal.setButtonGranted(button, '✅ Camera Access Granted');
            portal.updatePermissionStatus('cameraSetupStatus', 'Camera ready for photo capture');
            portal.updatePermissionCard('cameraSetupCard', 'granted');
            portal.updateSetupProgress();
            
            portal.showNotification('📷 Camera access granted! You can now attach photos to emergency reports.', 'success');
        })
        .catch(error => {
            portal.setButtonDenied(button, '❌ Camera Access Denied');
            portal.updatePermissionStatus('cameraSetupStatus', 'Camera access denied');
            portal.showNotification('❌ Camera access denied. Photo features will be unavailable.', 'error');
            
            setTimeout(() => portal.resetButton(button, 'Enable Camera Access'), 4000);
        });
};

portal.requestMicrophoneSetup = function(button) {
    portal.setButtonRequesting(button, 'Requesting microphone access...');
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            // Stop the stream immediately after getting permission
            stream.getTracks().forEach(track => track.stop());
            
            portal.permissionsGranted.microphone = true;
            portal.setButtonGranted(button, '✅ Voice Access Granted');
            portal.updatePermissionStatus('voiceSetupStatus', 'Microphone ready for voice reporting');
            portal.updatePermissionCard('voiceSetupCard', 'granted');
            portal.updateSetupProgress();
            
            portal.showNotification('🎤 Microphone access granted! Voice reporting is now available.', 'success');
        })
        .catch(error => {
            portal.setButtonDenied(button, '❌ Voice Access Denied');
            portal.updatePermissionStatus('voiceSetupStatus', 'Microphone access denied');
            portal.showNotification('❌ Microphone access denied. Voice features will be unavailable.', 'error');
            
            setTimeout(() => portal.resetButton(button, 'Enable Voice Access'), 4000);
        });
};

portal.requestNotificationsSetup = function(button) {
    if (!('Notification' in window)) {
        portal.showPermissionError(button, 'Notifications not supported by this browser');
        return;
    }
    
    portal.setButtonRequesting(button, 'Requesting notification access...');
    
    Notification.requestPermission().then(permission => {
        if (permission === 'granted') {
            portal.permissionsGranted.notifications = true;
            portal.setButtonGranted(button, '✅ Notifications Enabled');
            portal.updatePermissionStatus('notificationsSetupStatus', 'Notifications enabled for alerts');
            portal.updatePermissionCard('notificationsSetupCard', 'granted');
            portal.updateSetupProgress();
            
            // Show test notification
            new Notification('Emergency Response Assistant', { 
                body: '🔔 Notifications enabled! You\'ll receive important alerts.',
                icon: '/static/icons/notification-icon.png'
            });
            
            portal.showNotification('🔔 Notifications enabled! You\'ll receive emergency alerts and report updates.', 'success');
        } else {
            portal.setButtonDenied(button, '❌ Notifications Denied');
            portal.updatePermissionStatus('notificationsSetupStatus', 'Notification access denied');
            portal.showNotification('❌ Notification access denied. You won\'t receive alerts.', 'error');
            
            setTimeout(() => portal.resetButton(button, 'Enable Notifications'), 4000);
        }
    });
};

// Professional button state management
portal.setButtonRequesting = function(button, text) {
    button.textContent = text;
    button.disabled = true;
    button.className = 'btn btn-primary btn-permission requesting';
};

portal.setButtonGranted = function(button, text) {
    button.textContent = text;
    button.disabled = true;
    button.className = 'btn btn-primary btn-permission granted';
};

portal.setButtonDenied = function(button, text) {
    button.textContent = text;
    button.disabled = true;
    button.className = 'btn btn-primary btn-permission denied';
};

portal.resetButton = function(button, text) {
    button.textContent = text;
    button.disabled = false;
    button.className = 'btn btn-primary btn-permission';
};

// Professional permission status updates
portal.updatePermissionStatus = function(elementId, message) {
    const statusEl = document.getElementById(elementId);
    if (statusEl) {
        statusEl.textContent = message;
        statusEl.className = 'permission-status granted';
    }
};

portal.updatePermissionCard = function(cardId, status) {
    const card = document.getElementById(cardId);
    if (card) {
        card.classList.add(status);
    }
};

// Professional setup progress tracking
portal.updateSetupProgress = function() {
    const permissions = portal.permissionsGranted;
    const granted = Object.values(permissions).filter(p => p).length;
    const total = Object.keys(permissions).length;
    
    portal.setupProgress = (granted / total) * 100;
    
    // Update progress bar
    const progressFill = document.getElementById('progressFillSetup');
    if (progressFill) {
        progressFill.style.width = `${portal.setupProgress}%`;
    }
    
    // Update counter
    const progressCounter = document.getElementById('progressCounter');
    if (progressCounter) {
        progressCounter.textContent = `${granted} of ${total} permissions configured`;
    }
    
    // Update status text
    const setupStatus = document.getElementById('setupStatus');
    const statusText = setupStatus?.querySelector('.status-text');
    if (statusText) {
        if (granted === total) {
            statusText.textContent = '🎉 Setup complete! All permissions granted.';
            statusText.className = 'status-text complete';
            portal.setupComplete = true;
        } else if (granted > 0) {
            statusText.textContent = `${granted} permission${granted > 1 ? 's' : ''} granted. ${total - granted} remaining.`;
            statusText.className = 'status-text';
        } else {
            statusText.textContent = 'Configure permissions to get started';
            statusText.className = 'status-text';
        }
    }
    
    // Enable/disable complete setup button
    const completeBtn = document.getElementById('completeSetupBtn');
    if (completeBtn) {
        if (granted >= 2) { // At least 2 permissions (location + one other)
            completeBtn.disabled = false;
            completeBtn.textContent = granted === total ? 'Continue to Platform 🚀' : 'Continue with Current Setup →';
        } else {
            completeBtn.disabled = true;
            completeBtn.textContent = 'Complete Setup First';
        }
    }
    
    // Save progress
    localStorage.setItem('portalSetupProgress', JSON.stringify(portal.permissionsGranted));
};

// Emergency override functions
portal.emergencySkipSetup = function() {
    if (confirm('🚨 EMERGENCY OVERRIDE\n\nThis will skip setup and go directly to emergency reporting.\n\nSome features may not work without permissions.\n\nContinue?')) {
        portal.hasEmergencyOverride = true;
        portal.showNotification('🚨 Emergency mode activated. Going to report form...', 'warning');
        setTimeout(() => portal.goToStep(2), 1000);
    }
};

portal.skipSetup = function() {
    if (confirm('⚠️ Skip setup?\n\nWithout permissions, features like GPS location, photos, and voice reporting won\'t work.\n\nYou can complete setup later. Continue?')) {
        portal.showNotification('⚠️ Setup skipped. You can complete it later in settings.', 'warning');
        setTimeout(() => portal.goToStep(1), 1000);
    }
};

portal.completeSetup = function() {
    const granted = Object.values(portal.permissionsGranted).filter(p => p).length;
    
    if (granted >= 2) {
        portal.setupComplete = true;
        localStorage.setItem('portalSetupComplete', 'true');
        
        portal.showNotification('🎉 Setup complete! Welcome to Emergency Response Assistant.', 'success');
        setTimeout(() => portal.goToStep(1), 1000);
    } else {
        portal.showNotification('⚠️ Please grant at least 2 permissions to continue.', 'warning');
    }
};

// Enhanced notification system
portal.showNotification = function(message, type = 'info', duration = 4000) {
    const notification = document.createElement('div');
    notification.className = `portal-notification portal-notification-${type}`;
    
    const colors = {
        'success': '#22c55e',
        'warning': '#f59e0b',
        'error': '#ef4444',
        'info': '#3b82f6'
    };
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        z-index: 10000;
        font-weight: 500;
        font-size: 0.9rem;
        transform: translateX(100%);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        max-width: 400px;
        line-height: 1.4;
        font-family: var(--font-family);
        border: 1px solid rgba(255,255,255,0.2);
    `;
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 10);
    
    // Auto remove
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, duration);
};

// Enhanced welcome stats display
portal.showWelcomeStats = function() {
    // Animate stats with realistic data
    const stats = [
        { id: 'responseTime', value: '< 2 min', animate: false },
        { id: 'reportsProcessed', value: 1247, animate: true },
        { id: 'aiAccuracy', value: '94%', animate: false },
        { id: 'onlineUsers', value: 156, animate: true }
    ];
    
    stats.forEach(stat => {
        const element = document.getElementById(stat.id);
        if (element && stat.animate) {
            portal.animateNumber(element, 0, parseInt(stat.value), 2000);
        } else if (element) {
            element.textContent = stat.value;
        }
    });
};

// Professional number animation
portal.animateNumber = function(element, start, end, duration) {
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + (end - start) * easeOutCubic);
        
        element.textContent = current.toLocaleString();
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
};

// Enhanced report form initialization
portal.initializeReportForm = function() {
    const form = document.getElementById('emergencyReportForm');
    if (!form) return;
    
    // Create form fields based on selected method
    const selectedMethod = document.querySelector('.method-card.selected')?.dataset.method || 'text';
    
    let formHTML = '';
    
    switch(selectedMethod) {
        case 'text':
            formHTML = `
                <div class="form-group">
                    <label class="form-label">📝 Emergency Description</label>
                    <textarea id="emergencyDescription" 
                              placeholder="Describe the emergency situation in detail..."
                              style="width: 100%; min-height: 120px; padding: 1rem; border: 2px solid var(--border-normal); border-radius: 8px; font-family: var(--font-family); font-size: var(--text-body2); resize: vertical;"
                              oninput="portal.analyzeText(this.value)"></textarea>
                    <div class="form-text">Be as specific as possible about location, injuries, and immediate dangers.</div>
                </div>
                <div class="form-group">
                    <label class="form-label">⚠️ Severity Level</label>
                    <select id="severityLevel" style="width: 100%; padding: 0.75rem; border: 2px solid var(--border-normal); border-radius: 8px; font-family: var(--font-family);">
                        <option value="low">🟢 Low - Non-urgent situation</option>
                        <option value="medium">🟡 Medium - Needs attention</option>
                        <option value="high">🟠 High - Urgent response needed</option>
                        <option value="critical">🔴 Critical - Life threatening</option>
                    </select>
                </div>
            `;
            break;
            
        case 'photo':
            formHTML = `
                <div class="form-group">
                    <label class="form-label">📸 Photo Evidence</label>
                    <input type="file" id="photoInput" accept="image/*" capture="environment"
                           style="width: 100%; padding: 1rem; border: 2px dashed var(--border-normal); border-radius: 8px; background: var(--bg-secondary);"
                           onchange="portal.handlePhotoUpload(this)">
                    <div class="form-text">Take or upload a photo showing the emergency situation.</div>
                </div>
                <div id="photoPreview" style="display: none; margin-top: 1rem;">
                    <img id="previewImage" style="max-width: 100%; max-height: 300px; border-radius: 8px; box-shadow: var(--shadow-md);">
                    <div id="aiPhotoAnalysis" style="margin-top: 1rem; padding: 1rem; background: var(--success-light); border-radius: 8px; border: 1px solid var(--green-200);">
                        <h4 style="color: var(--green-700); margin-bottom: 0.5rem;">🤖 AI Photo Analysis</h4>
                        <div id="photoAnalysisResults">Analyzing image...</div>
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">📝 Additional Details (Optional)</label>
                    <textarea id="photoDescription" 
                              placeholder="Add any additional context about the photo..."
                              style="width: 100%; min-height: 80px; padding: 1rem; border: 2px solid var(--border-normal); border-radius: 8px; font-family: var(--font-family);"></textarea>
                </div>
            `;
            break;
            
        case 'location':
            formHTML = `
                <div class="form-group">
                    <label class="form-label">📍 Emergency Location</label>
                    <div id="locationPicker" style="width: 100%; height: 200px; border: 2px solid var(--border-normal); border-radius: 8px; background: var(--bg-secondary); display: flex; align-items: center; justify-content: center; color: var(--text-secondary);">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📍</div>
                            <div>Click to set emergency location</div>
                            <button type="button" onclick="portal.getCurrentLocation()" style="margin-top: 1rem; padding: 0.5rem 1rem; background: var(--primary); color: white; border: none; border-radius: 6px; cursor: pointer;">Use Current Location</button>
                        </div>
                    </div>
                    <div class="form-text" id="selectedLocation">No location selected</div>
                </div>
                <div class="form-group">
                    <label class="form-label">📝 What's happening at this location?</label>
                    <textarea id="locationDescription" 
                              placeholder="Describe the emergency at this location..."
                              style="width: 100%; min-height: 100px; padding: 1rem; border: 2px solid var(--border-normal); border-radius: 8px; font-family: var(--font-family);"></textarea>
                </div>
            `;
            break;
    }
    
    form.innerHTML = formHTML;
};

// Enhanced method selection
portal.selectMethod = function(element, method) {
    // Remove selected class from all cards
    const cards = document.querySelectorAll('.method-card');
    cards.forEach(c => c.classList.remove('selected'));
    
    // Add selected class to clicked card
    element.classList.add('selected');
    
    // Reinitialize form for new method
    portal.initializeReportForm();
    
    // Show notification about method selection
    const methodNames = {
        'text': 'Text reporting',
        'photo': 'Photo evidence',
        'location': 'Location-based reporting'
    };
    
    portal.showNotification(`📱 ${methodNames[method]} selected. Form updated.`, 'info', 2000);
};

// AI text analysis simulation
portal.analyzeText = function(text) {
    if (text.length < 10) return;
    
    const analysisDiv = document.getElementById('realtimeAnalysis');
    if (!analysisDiv) return;
    
    analysisDiv.style.display = 'block';
    
    // Simulate AI analysis with realistic delays
    setTimeout(() => {
        const severityEl = document.getElementById('severityLevel');
        if (severityEl) {
            // Simple keyword-based severity detection
            const highSeverityWords = ['fire', 'blood', 'unconscious', 'trapped', 'explosion', 'accident'];
            const mediumSeverityWords = ['injured', 'smoke', 'flooding', 'broken', 'emergency'];
            
            const textLower = text.toLowerCase();
            let detectedSeverity = 'low';
            
            if (highSeverityWords.some(word => textLower.includes(word))) {
                detectedSeverity = 'high';
            } else if (mediumSeverityWords.some(word => textLower.includes(word))) {
                detectedSeverity = 'medium';
            }
            
            document.getElementById('severityLevel').textContent = `AI suggests: ${detectedSeverity.toUpperCase()}`;
        }
        
        document.getElementById('responseType').textContent = 'Emergency Services';
        document.getElementById('estimatedETA').textContent = '8-12 minutes';
    }, 800);
};

// Photo upload and analysis
portal.handlePhotoUpload = function(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const preview = document.getElementById('photoPreview');
            const image = document.getElementById('previewImage');
            
            image.src = e.target.result;
            preview.style.display = 'block';
            
            // Simulate AI photo analysis
            portal.analyzePhoto(file);
        };
        
        reader.readAsDataURL(file);
    }
};

portal.analyzePhoto = function(file) {
    const resultsDiv = document.getElementById('photoAnalysisResults');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = 'Analyzing image...';
    
    // Simulate AI analysis
    setTimeout(() => {
        const analyses = [
            '🔥 Fire/smoke detected in image',
            '🚗 Vehicle accident visible',
            '🏢 Structural damage identified',
            '🌊 Flooding conditions present',
            '👥 Multiple people visible in scene',
            '⚡ Hazardous conditions detected'
        ];
        
        const randomAnalysis = analyses[Math.floor(Math.random() * analyses.length)];
        
        resultsDiv.innerHTML = `
            <div style="margin-bottom: 0.5rem;">${randomAnalysis}</div>
            <div style="color: var(--green-600); font-size: 0.9rem;">📊 Confidence: 87% | 🎯 Severity: High | ⏰ Response: Immediate</div>
        `;
    }, 2000);
};

// Location handling
portal.getCurrentLocation = function() {
    if (!navigator.geolocation) {
        portal.showNotification('❌ Geolocation not supported', 'error');
        return;
    }
    
    portal.showNotification('📍 Getting your location...', 'info', 2000);
    
    navigator.geolocation.getCurrentPosition(
        (position) => {
            const { latitude, longitude } = position.coords;
            
            document.getElementById('selectedLocation').textContent = 
                `Location set: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
            
            portal.userLocation = { lat: latitude, lng: longitude };
            portal.showNotification('✅ Current location set successfully', 'success');
        },
        (error) => {
            portal.showNotification('❌ Could not get location: ' + error.message, 'error');
        }
    );
};

// Enhanced report submission
portal.submitReport = function() {
    const selectedMethod = document.querySelector('.method-card.selected')?.dataset.method || 'text';
    
    // Collect form data based on method
    let reportData = {
        method: selectedMethod,
        timestamp: new Date().toISOString(),
        location: portal.userLocation
    };
    
    switch(selectedMethod) {
        case 'text':
            reportData.description = document.getElementById('emergencyDescription')?.value;
            reportData.severity = document.getElementById('severityLevel')?.value;
            break;
        case 'photo':
            reportData.photo = document.getElementById('previewImage')?.src;
            reportData.description = document.getElementById('photoDescription')?.value;
            break;
        case 'location':
            reportData.description = document.getElementById('locationDescription')?.value;
            break;
    }
    
    // Validate required fields
    if (!reportData.description || reportData.description.trim().length < 10) {
        portal.showNotification('⚠️ Please provide a detailed description (at least 10 characters)', 'warning');
        return;
    }
    
    // Save report to localStorage
    const existingReports = JSON.parse(localStorage.getItem('userReports') || '[]');
    reportData.id = Date.now();
    reportData.status = 'submitted';
    existingReports.push(reportData);
    localStorage.setItem('userReports', JSON.stringify(existingReports));
    
    // Show success and redirect
    portal.showNotification('📤 Emergency report submitted successfully! Report ID: #' + reportData.id, 'success');
    setTimeout(() => portal.goToStep(4), 2000);
};

// Enhanced voice recording
portal.toggleRecording = function() {
    const button = document.getElementById('recordButton');
    const status = document.getElementById('voiceStatus');
    const transcriptBox = document.getElementById('transcriptBox');
    
    if (portal.isVoiceRecording) {
        // Stop recording
        portal.isVoiceRecording = false;
        button.textContent = '🎤';
        button.classList.remove('recording');
        status.textContent = 'Recording stopped. Processing transcript...';
        
        // Simulate processing and show analysis
        setTimeout(() => {
            status.textContent = 'Recording complete. Review transcript and submit when ready.';
            
            const analysisDiv = document.getElementById('voiceAnalysis');
            if (analysisDiv) {
                analysisDiv.style.display = 'block';
                
                // Simulate AI analysis results
                setTimeout(() => {
                    document.getElementById('urgencyLevel').textContent = 'HIGH - Immediate response needed';
                    document.getElementById('emotionLevel').textContent = 'Distressed - Speaker sounds panicked';
                    document.getElementById('detectedLocation').textContent = 'Main Street mentioned';
                    document.getElementById('aiRecommendation').textContent = 'Dispatch emergency services immediately';
                }, 1000);
            }
            
            // Enable submit button
            const submitBtn = document.getElementById('submitVoiceReport');
            if (submitBtn) {
                submitBtn.disabled = false;
            }
        }, 2000);
        
    } else {
        // Start recording
        if (!portal.permissionsGranted.microphone) {
            portal.showNotification('🎤 Please enable microphone access first', 'warning');
            portal.goToStep(0);
            return;
        }
        
        portal.isVoiceRecording = true;
        button.textContent = '⏹️';
        button.classList.add('recording');
        status.textContent = '🎤 Recording... Speak clearly about the emergency situation';
        
        // Clear previous transcript
        transcriptBox.innerHTML = '<div class="transcript-placeholder">Listening...</div>';
        transcriptBox.classList.add('active');
        
        // Simulate real-time transcript after 3 seconds
        setTimeout(() => {
            if (portal.isVoiceRecording) {
                transcriptBox.innerHTML = 'There\'s a serious car accident on Main Street near the intersection with Oak Avenue. Multiple vehicles involved, people appear to be injured and trapped. We need emergency services here immediately. There\'s smoke coming from one of the cars and traffic is completely blocked.';
                portal.voiceTranscript = transcriptBox.textContent;
            }
        }, 3000);
    }
};

portal.submitVoiceReport = function() {
    if (!portal.voiceTranscript) {
        portal.showNotification('⚠️ No voice transcript available', 'warning');
        return;
    }
    
    // Save voice report
    const reportData = {
        method: 'voice',
        transcript: portal.voiceTranscript,
        timestamp: new Date().toISOString(),
        location: portal.userLocation,
        id: Date.now(),
        status: 'submitted'
    };
    
    const existingReports = JSON.parse(localStorage.getItem('userReports') || '[]');
    existingReports.push(reportData);
    localStorage.setItem('userReports', JSON.stringify(existingReports));
    
    portal.showNotification('🎤 Voice report submitted successfully! Report ID: #' + reportData.id, 'success');
    setTimeout(() => portal.goToStep(4), 2000);
};

// Enhanced reports loading with professional display
portal.loadUserReports = function() {
    const grid = document.getElementById('reportsGrid');
    if (!grid) return;
    
    const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
    
    // Update dashboard stats
    portal.updateDashboardStats(savedReports);
    
    if (savedReports.length === 0) {
        grid.innerHTML = `
            <div style="text-align: center; padding: 3rem; color: var(--text-secondary); grid-column: 1 / -1;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
                <h3 style="color: var(--text-primary); margin-bottom: 0.5rem;">No Reports Yet</h3>
                <p style="margin-bottom: 2rem;">You haven't submitted any emergency reports yet.</p>
                <button class="btn btn-primary" onclick="portal.goToStep(2)" style="margin-right: 1rem;">
                    📱 Submit Your First Report
                </button>
                <button class="btn btn-outline" onclick="portal.goToStep(0)">
                    🛠️ Complete Setup First
                </button>
            </div>
        `;
    } else {
        const reportsHTML = savedReports.map((report, index) => {
            const timeAgo = portal.getTimeAgo(new Date(report.timestamp));
            const statusIcon = report.status === 'submitted' ? '⏳' : '✅';
            const methodIcon = {
                'text': '📝',
                'voice': '🎤',
                'photo': '📸',
                'location': '📍'
            }[report.method] || '📝';
            
            return `
                <div style="background: white; border: 2px solid var(--border-light); border-radius: 12px; padding: 1.5rem; box-shadow: var(--shadow-sm); transition: all 0.3s ease; cursor: pointer;" 
                     onclick="portal.viewReportDetails(${report.id})"
                     onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='var(--shadow-md)'"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='var(--shadow-sm)'">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1.25rem;">${methodIcon}</span>
                            <h4 style="margin: 0; color: var(--text-primary); font-size: var(--text-body1);">Report #${report.id}</h4>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                            ${statusIcon} ${report.status}
                        </div>
                    </div>
                    <p style="margin: 0 0 1rem 0; color: var(--text-secondary); line-height: 1.5;">
                        ${(report.description || report.transcript || 'Emergency report').substring(0, 100)}${(report.description || report.transcript || '').length > 100 ? '...' : ''}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.875rem; color: var(--text-muted);">
                        <span>📍 ${report.location ? `${report.location.lat.toFixed(4)}, ${report.location.lng.toFixed(4)}` : 'No location'}</span>
                        <span>⏰ ${timeAgo}</span>
                    </div>
                </div>
            `;
        }).join('');
        
        grid.innerHTML = reportsHTML;
    }
};

portal.updateDashboardStats = function(reports) {
    const total = reports.length;
    const pending = reports.filter(r => r.status === 'submitted').length;
    const resolved = reports.filter(r => r.status === 'resolved').length;
    
    document.getElementById('totalReports').textContent = total;
    document.getElementById('pendingReports').textContent = pending;
    document.getElementById('resolvedReports').textContent = resolved;
    document.getElementById('avgResponseTime').textContent = total > 0 ? '8 min' : '--';
};

portal.getTimeAgo = function(date) {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
    if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    return 'Just now';
};

portal.viewReportDetails = function(reportId) {
    const reports = JSON.parse(localStorage.getItem('userReports') || '[]');
    const report = reports.find(r => r.id === reportId);
    
    if (report) {
        portal.showNotification(`📋 Report #${reportId} details - Feature coming soon!`, 'info');
    }
};

// Enhanced quick emergency function
portal.quickEmergency = function() {
    if (confirm('🚨 Quick Emergency Report\n\nThis will take you directly to the emergency report form.\n\nFor life-threatening emergencies, call 911 first!\n\nContinue?')) {
        portal.hasEmergencyOverride = true;
        portal.goToStep(2);
    }
};

// Legacy compatibility functions
portal.requestLocation = portal.requestLocationSetup;
portal.requestMicrophone = portal.requestMicrophoneSetup;
portal.requestCamera = portal.requestCameraSetup;
portal.requestNotifications = portal.requestNotificationsSetup;

// Enhanced initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Enhanced Portal Initialized');
    
    // Check for existing setup
    const savedPermissions = localStorage.getItem('portalSetupProgress');
    if (savedPermissions) {
        portal.permissionsGranted = JSON.parse(savedPermissions);
        portal.updateSetupProgress();
    }
    
    const setupComplete = localStorage.getItem('portalSetupComplete');
    if (setupComplete === 'true') {
        portal.setupComplete = true;
        // Start on welcome page if setup is complete
        portal.goToStep(1);
    } else {
        // Start on setup page for new users
        portal.goToStep(0);
    }
    
    // Enhanced keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 'e':
                    e.preventDefault();
                    portal.quickEmergency();
                    break;
                case 'v':
                    e.preventDefault();
                    portal.goToStep(3);
                    break;
                case 's':
                    e.preventDefault();
                    portal.goToStep(0);
                    break;
            }
        }
    });
    
    console.log('✅ Enhanced Portal Ready');
});

// Make portal globally available
window.portal = portal;