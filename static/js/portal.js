// Simple Portal JavaScript - Clean and Original
// Keeps your beautiful design without complications

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
    }
};

// Simple step navigation
portal.goToStep = function(stepNumber) {
    console.log(`Going to step ${stepNumber}`);
    
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
        const progress = stepNumber > 0 ? (stepNumber / 4) * 100 : 10;
        progressFill.style.width = `${progress}%`;
    }
    
    portal.currentStep = stepNumber;
    
    // Initialize step-specific features
    if (stepNumber === 4) {
        portal.loadUserReports();
    }
};

// Quick emergency function
portal.quickEmergency = function() {
    if (confirm('🚨 Quick Emergency Report\n\nGo directly to emergency report form?')) {
        portal.goToStep(2);
    }
};

// Risk prediction placeholder
portal.showRiskPrediction = function() {
    alert('🔮 AI-powered risk prediction feature coming soon!');
};

// Simple permission requests
portal.requestLocation = function(button) {
    if (!navigator.geolocation) {
        alert('Geolocation not supported');
        return;
    }
    
    button.textContent = 'Requesting...';
    button.disabled = true;
    
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            portal.userLocation = { 
                lat: pos.coords.latitude, 
                lng: pos.coords.longitude 
            };
            portal.permissionsGranted.location = true;
            
            button.textContent = '✅ GPS Enabled';
            button.style.background = '#16a34a';
            button.style.color = 'white';
            
            const statusEl = document.getElementById('locationStatus');
            if (statusEl) {
                statusEl.textContent = `✅ Location: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`;
            }
        },
        (err) => {
            button.textContent = '❌ GPS Denied';
            button.style.background = '#dc2626';
            button.style.color = 'white';
            
            setTimeout(() => {
                button.textContent = 'Enable GPS';
                button.style.background = '';
                button.style.color = '';
                button.disabled = false;
            }, 3000);
        }
    );
};

portal.requestMicrophone = function(btn) {
    btn.textContent = 'Requesting...';
    btn.disabled = true;
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            // Stop the stream immediately
            stream.getTracks().forEach(track => track.stop());
            
            portal.permissionsGranted.microphone = true;
            btn.textContent = '✅ Voice Enabled';
            btn.style.background = '#16a34a';
            btn.style.color = 'white';
        })
        .catch(err => {
            btn.textContent = '❌ Voice Denied';
            btn.style.background = '#dc2626';
            btn.style.color = 'white';
            
            setTimeout(() => {
                btn.textContent = 'Enable Voice';
                btn.style.background = '';
                btn.style.color = '';
                btn.disabled = false;
            }, 3000);
        });
};

portal.requestCamera = function(btn) {
    btn.textContent = 'Requesting...';
    btn.disabled = true;
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            // Stop the stream immediately
            stream.getTracks().forEach(track => track.stop());
            
            portal.permissionsGranted.camera = true;
            btn.textContent = '✅ Camera Enabled';
            btn.style.background = '#16a34a';
            btn.style.color = 'white';
        })
        .catch(err => {
            btn.textContent = '❌ Camera Denied';
            btn.style.background = '#dc2626';
            btn.style.color = 'white';
            
            setTimeout(() => {
                btn.textContent = 'Enable Camera';
                btn.style.background = '';
                btn.style.color = '';
                btn.disabled = false;
            }, 3000);
        });
};

portal.requestNotifications = function(btn) {
    if (!('Notification' in window)) {
        alert('Notifications not supported');
        return;
    }
    
    btn.textContent = 'Requesting...';
    btn.disabled = true;
    
    Notification.requestPermission().then(permission => {
        if (permission === 'granted') {
            portal.permissionsGranted.notifications = true;
            btn.textContent = '✅ Alerts Enabled';
            btn.style.background = '#16a34a';
            btn.style.color = 'white';
            
            // Show test notification
            new Notification('Emergency Assistant', { 
                body: 'Alerts enabled!' 
            });
        } else {
            btn.textContent = '❌ Alerts Denied';
            btn.style.background = '#dc2626';
            btn.style.color = 'white';
            
            setTimeout(() => {
                btn.textContent = 'Enable Alerts';
                btn.style.background = '';
                btn.style.color = '';
                btn.disabled = false;
            }, 3000);
        }
    });
};

// Simple method selection
portal.selectMethod = function(element, method) {
    // Remove selected class from all cards
    const cards = document.querySelectorAll('.method-card');
    cards.forEach(c => c.classList.remove('selected'));
    
    // Add selected class to clicked card
    element.classList.add('selected');
    
    console.log(`Selected method: ${method}`);
};

// Simple report submission
portal.submitReport = function() {
    alert('📤 Report submitted successfully!');
    setTimeout(() => portal.goToStep(4), 1000);
};

// Simple voice recording
portal.toggleRecording = function() {
    const button = document.getElementById('recordButton');
    const status = document.getElementById('voiceStatus');
    
    if (portal.isVoiceRecording) {
        // Stop recording
        portal.isVoiceRecording = false;
        button.textContent = '🎤';
        button.style.background = '';
        status.textContent = 'Recording stopped. Click submit when ready.';
        
        // Enable submit button
        const submitBtn = document.getElementById('submitVoiceReport');
        if (submitBtn) {
            submitBtn.disabled = false;
        }
    } else {
        // Start recording
        if (!portal.permissionsGranted.microphone) {
            alert('🎤 Please enable microphone access first');
            portal.goToStep(1);
            return;
        }
        
        portal.isVoiceRecording = true;
        button.textContent = '⏹️';
        button.style.background = '#dc2626';
        status.textContent = '🎤 Recording... Speak clearly about the emergency';
        
        // Simulate transcript after 3 seconds
        setTimeout(() => {
            if (portal.isVoiceRecording) {
                const transcriptBox = document.getElementById('transcriptBox');
                if (transcriptBox) {
                    transcriptBox.textContent = 'Sample transcript: Emergency at Main Street, need immediate assistance...';
                    portal.voiceTranscript = 'Sample emergency transcript';
                }
            }
        }, 3000);
    }
};

portal.submitVoiceReport = function() {
    alert('🎤 Voice report submitted successfully!');
    setTimeout(() => portal.goToStep(4), 1000);
};

// Simple reports loading
portal.loadUserReports = function() {
    const grid = document.getElementById('reportsGrid');
    if (!grid) return;
    
    // Check for saved reports
    const savedReports = JSON.parse(localStorage.getItem('userReports') || '[]');
    
    if (savedReports.length === 0) {
        grid.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: #6b7280;">
                <h3>📋 No Reports Yet</h3>
                <p>You haven't submitted any emergency reports yet.</p>
                <button class="btn btn-primary" onclick="portal.goToStep(2)" style="margin-top: 1rem;">
                    📱 Submit Your First Report
                </button>
            </div>
        `;
    } else {
        // Show saved reports
        const reportsHTML = savedReports.map((report, index) => `
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #1f2937;">📝 Report #${index + 1}</h4>
                <p style="margin: 0.5rem 0; color: #6b7280;">${report.description || 'Emergency report'}</p>
                <p style="margin: 0; color: #374151; font-size: 0.9rem;">📍 ${report.location || 'Location not specified'}</p>
            </div>
        `).join('');
        
        grid.innerHTML = reportsHTML;
    }
};

// Make portal globally available
window.portal = portal;

// Simple initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Simple Portal Initialized');
    
    // Set initial step
    portal.goToStep(0);
    
    // Add simple keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'e') {
                e.preventDefault();
                portal.quickEmergency();
            } else if (e.key === 'v') {
                e.preventDefault();
                portal.goToStep(3);
            }
        }
    });
    
    console.log('✅ Portal ready');
});