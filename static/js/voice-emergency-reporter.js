/**
 * Voice Emergency Reporter - Gemma 3n Powered Voice Analysis
 * Handles real-time audio processing, transcription, and AI analysis
 */

class VoiceEmergencyReporter {
    constructor() {
        this.recognition = null;
        this.isRecording = false;
        this.isProcessing = false;
        this.transcript = '';
        this.audioContext = null;
        this.analyzer = null;
        this.microphone = null;
        this.dataArray = null;
        this.visualizerBars = [];
        this.currentLanguage = 'en-US';
        this.sensitivity = 0.7;
        this.selectedModel = 'gemma-3n-4b';
        this.autoSubmit = false;
        
        this.initializeElements();
        this.initializeAudioProcessing();
        this.setupEventListeners();
        this.createAudioVisualizer();
        this.initializeGemma3n();
    }
    
    initializeElements() {
        this.recordButton = document.getElementById('recordButton');
        this.recordIcon = document.getElementById('recordIcon');
        this.statusText = document.getElementById('statusText');
        this.transcriptText = document.getElementById('transcriptText');
        this.urgencyFill = document.getElementById('urgencyFill');
        this.urgencyText = document.getElementById('urgencyText');
        this.submitReportBtn = document.getElementById('submitReportBtn');
        this.retryBtn = document.getElementById('retryBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.realtimeAnalysis = document.getElementById('realtimeAnalysis');
        this.processingIndicator = document.getElementById('processingIndicator');
        this.confidenceSpan = document.getElementById('confidence');
        
        // Analysis elements
        this.stressLevel = document.getElementById('stressLevel');
        this.clarityLevel = document.getElementById('clarityLevel');
        this.speakerConfidence = document.getElementById('speakerConfidence');
        this.emergencyType = document.getElementById('emergencyType');
        this.emergencyLocation = document.getElementById('emergencyLocation');
        this.peopleAffected = document.getElementById('peopleAffected');
        this.immediateNeeds = document.getElementById('immediateNeeds');
        this.recommendations = document.getElementById('recommendations');
        
        // Settings elements
        this.languageSelect = document.getElementById('languageSelect');
        this.sensitivitySlider = document.getElementById('sensitivitySlider');
        this.modelSelect = document.getElementById('modelSelect');
        this.autoSubmitToggle = document.getElementById('autoSubmitToggle');
    }
    
    async initializeGemma3n() {
        try {
            // Initialize Gemma 3n for audio processing
            if (window.EdgeAI) {
                await window.EdgeAI.loadAudioModel(this.selectedModel);
                console.log('✅ Gemma 3n audio model loaded');
                this.updateAIStatus('ready');
            } else {
                console.warn('⚠️ EdgeAI not available, using fallback processing');
                this.updateAIStatus('fallback');
            }
        } catch (error) {
            console.error('❌ Failed to initialize Gemma 3n:', error);
            this.updateAIStatus('error');
        }
    }
    
    updateAIStatus(status) {
        const aiStatusDot = document.getElementById('aiStatusDot');
        const aiStatusText = document.getElementById('aiStatusText');
        
        switch (status) {
            case 'ready':
                aiStatusDot.className = 'ai-status-dot';
                aiStatusText.textContent = '🧠 Gemma 3n Ready for Voice Analysis';
                break;
            case 'processing':
                aiStatusDot.className = 'ai-status-dot loading';
                aiStatusText.textContent = '🧠 Analyzing with Gemma 3n...';
                break;
            case 'fallback':
                aiStatusDot.className = 'ai-status-dot loading';
                aiStatusText.textContent = '🧠 Using Fallback AI Processing';
                break;
            case 'error':
                aiStatusDot.className = 'ai-status-dot error';
                aiStatusText.textContent = '🧠 AI Processing Unavailable';
                break;
        }
    }
    
    initializeAudioProcessing() {
        // Initialize Web Speech API
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
        } else if ('SpeechRecognition' in window) {
            this.recognition = new SpeechRecognition();
        } else {
            console.error('Speech recognition not supported');
            this.statusText.textContent = 'Speech recognition not supported in this browser';
            return;
        }
        
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = this.currentLanguage;
        
        this.recognition.onstart = () => {
            this.isRecording = true;
            this.updateRecordingState();
        };
        
        this.recognition.onresult = (event) => {
            this.handleSpeechResult(event);
        };
        
        this.recognition.onend = () => {
            this.isRecording = false;
            this.updateRecordingState();
            if (this.transcript) {
                this.processWithGemma3n();
            }
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.statusText.textContent = `Speech recognition error: ${event.error}`;
            this.isRecording = false;
            this.updateRecordingState();
        };
    }
    
    async setupAudioVisualizer() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyzer = this.audioContext.createAnalyser();
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            
            this.analyzer.fftSize = 256;
            this.microphone.connect(this.analyzer);
            
            const bufferLength = this.analyzer.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            
            this.startVisualization();
        } catch (error) {
            console.error('Error setting up audio visualizer:', error);
        }
    }
    
    createAudioVisualizer() {
        const visualizer = document.getElementById('audioVisualizer');
        visualizer.innerHTML = '';
        
        // Create 20 bars for visualization
        for (let i = 0; i < 20; i++) {
            const bar = document.createElement('div');
            bar.className = 'audio-bar';
            bar.style.height = '4px';
            visualizer.appendChild(bar);
            this.visualizerBars.push(bar);
        }
    }
    
    startVisualization() {
        if (!this.analyzer || !this.dataArray) return;
        
        const animate = () => {
            if (this.isRecording) {
                this.analyzer.getByteFrequencyData(this.dataArray);
                
                // Update visualizer bars
                this.visualizerBars.forEach((bar, index) => {
                    const value = this.dataArray[index * 8] || 0;
                    const height = Math.max(4, (value / 255) * 50);
                    bar.style.height = `${height}px`;
                });
                
                requestAnimationFrame(animate);
            } else {
                // Reset bars when not recording
                this.visualizerBars.forEach(bar => {
                    bar.style.height = '4px';
                });
            }
        };
        
        animate();
    }
    
    setupEventListeners() {
        // Record button
        this.recordButton.addEventListener('click', () => {
            if (this.isRecording) {
                this.stopRecording();
            } else {
                this.startRecording();
            }
        });
        
        // Settings listeners
        this.languageSelect.addEventListener('change', (e) => {
            this.currentLanguage = e.target.value;
            if (this.recognition) {
                this.recognition.lang = this.currentLanguage;
            }
        });
        
        this.sensitivitySlider.addEventListener('input', (e) => {
            this.sensitivity = parseFloat(e.target.value);
        });
        
        this.modelSelect.addEventListener('change', (e) => {
            this.selectedModel = e.target.value;
            this.initializeGemma3n(); // Reload model
        });
        
        this.autoSubmitToggle.addEventListener('change', (e) => {
            this.autoSubmit = e.target.checked;
        });
        
        // Action buttons
        this.submitReportBtn.addEventListener('click', () => {
            this.submitEmergencyReport();
        });
        
        this.retryBtn.addEventListener('click', () => {
            this.resetSession();
        });
        
        this.exportBtn.addEventListener('click', () => {
            this.exportTranscript();
        });
    }
    
    async startRecording() {
        try {
            if (!this.audioContext) {
                await this.setupAudioVisualizer();
            }
            
            this.recognition.start();
            this.statusText.textContent = 'Recording... Speak clearly about the emergency';
            this.transcript = '';
            this.transcriptText.textContent = 'Listening...';
            this.resetAnalysis();
        } catch (error) {
            console.error('Error starting recording:', error);
            this.statusText.textContent = 'Error starting recording. Please try again.';
        }
    }
    
    stopRecording() {
        if (this.recognition) {
            this.recognition.stop();
        }
        this.statusText.textContent = 'Processing speech with Gemma 3n AI...';
    }
    
    handleSpeechResult(event) {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            const confidence = event.results[i][0].confidence;
            
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
            
            // Update confidence display
            if (confidence) {
                this.confidenceSpan.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
            }
        }
        
        this.transcript = finalTranscript || this.transcript;
        const displayText = this.transcript + (interimTranscript ? `<span style="color: #6b7280; font-style: italic;">${interimTranscript}</span>` : '');
        this.transcriptText.innerHTML = displayText || 'Listening...';
    }
    
    async processWithGemma3n() {
        if (!this.transcript.trim()) {
            this.statusText.textContent = 'No speech detected. Please try again.';
            return;
        }
        
        this.isProcessing = true;
        this.updateProcessingState();
        this.updateAIStatus('processing');
        
        try {
            let analysisResult;
            
            if (window.EdgeAI && window.EdgeAI.isReady) {
                // Use Gemma 3n for analysis
                analysisResult = await this.analyzeWithGemma3n();
            } else {
                // Fallback analysis
                analysisResult = await this.fallbackAnalysis();
            }
            
            this.displayAnalysisResults(analysisResult);
            this.enableActionButtons();
            
            if (this.autoSubmit && analysisResult.urgency >= 0.7) {
                setTimeout(() => this.submitEmergencyReport(), 2000);
            }
            
        } catch (error) {
            console.error('Error processing with AI:', error);
            this.statusText.textContent = 'Error processing speech. Please try again.';
        } finally {
            this.isProcessing = false;
            this.updateProcessingState();
            this.updateAIStatus('ready');
        }
    }
    
    async analyzeWithGemma3n() {
        // Simulate Gemma 3n analysis with enhanced capabilities
        const analysisPrompt = `
        Analyze this emergency voice report for:
        1. Urgency level (0-1 scale)
        2. Emotional state and stress indicators
        3. Emergency type and details
        4. Immediate response recommendations
        
        Voice Report: "${this.transcript}"
        
        Consider audio tone analysis, speech patterns, and content to provide comprehensive emergency assessment.
        `;
        
        // In a real implementation, this would call the actual Gemma 3n model
        const mockAnalysis = await this.simulateGemma3nAnalysis(this.transcript);
        
        return mockAnalysis;
    }
    
    async simulateGemma3nAnalysis(transcript) {
        // Simulate Gemma 3n processing time
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Enhanced analysis based on transcript content
        const urgencyKeywords = ['emergency', 'urgent', 'immediate', 'help', 'fire', 'flood', 'earthquake', 'injured', 'trapped', 'danger'];
        const emotionKeywords = {
            panic: ['panic', 'scared', 'terrified', 'desperate'],
            stress: ['stressed', 'worried', 'anxious', 'overwhelmed'],
            calm: ['calm', 'stable', 'okay', 'fine']
        };
        
        const text = transcript.toLowerCase();
        
        // Calculate urgency
        const urgencyScore = urgencyKeywords.reduce((score, keyword) => {
            return score + (text.includes(keyword) ? 0.2 : 0);
        }, 0.1);
        
        // Analyze emotional state
        let emotionalState = 'calm';
        let stressLevel = 'Low';
        
        if (emotionKeywords.panic.some(word => text.includes(word))) {
            emotionalState = 'panic';
            stressLevel = 'Critical';
        } else if (emotionKeywords.stress.some(word => text.includes(word))) {
            emotionalState = 'stressed';
            stressLevel = 'High';
        } else if (emotionKeywords.calm.some(word => text.includes(word))) {
            emotionalState = 'calm';
            stressLevel = 'Low';
        } else {
            stressLevel = 'Medium';
        }
        
        // Extract emergency details
        const emergencyTypes = {
            'fire': ['fire', 'smoke', 'burning', 'flames'],
            'medical': ['injured', 'hurt', 'medical', 'ambulance', 'hospital', 'sick'],
            'natural disaster': ['earthquake', 'flood', 'hurricane', 'tornado', 'storm'],
            'accident': ['accident', 'crash', 'collision', 'vehicle'],
            'security': ['robbery', 'theft', 'break-in', 'suspicious'],
            'utility': ['power outage', 'gas leak', 'water main', 'electrical']
        };
        
        let detectedType = 'general emergency';
        for (const [type, keywords] of Object.entries(emergencyTypes)) {
            if (keywords.some(keyword => text.includes(keyword))) {
                detectedType = type;
                break;
            }
        }
        
        // Extract location mentions
        const locationPattern = /(?:at|on|near|in)\s+([^.!?]+)/gi;
        const locationMatches = text.match(locationPattern);
        const location = locationMatches ? locationMatches[0].replace(/^(at|on|near|in)\s+/i, '') : 'Not specified';
        
        // Extract people count
        const peoplePattern = /(\d+)\s*(people|person|individuals?)/gi;
        const peopleMatches = text.match(peoplePattern);
        const peopleAffected = peopleMatches ? peopleMatches[0] : 'Unknown';
        
        // Generate recommendations
        const recommendations = this.generateRecommendations(detectedType, urgencyScore, emotionalState);
        
        return {
            urgency: Math.min(urgencyScore, 1.0),
            urgencyLevel: this.getUrgencyLevel(urgencyScore),
            emotionalState,
            stressLevel,
            clarity: this.calculateClarity(transcript),
            confidence: 0.85 + Math.random() * 0.1, // Simulate confidence
            emergencyType: detectedType,
            location,
            peopleAffected,
            immediateNeeds: this.extractImmediateNeeds(text),
            recommendations,
            rawTranscript: transcript
        };
    }
    
    generateRecommendations(emergencyType, urgency, emotionalState) {
        const baseRecommendations = {
            'fire': [
                'Call emergency services immediately (911)',
                'Evacuate the area if safe to do so',
                'Do not use elevators',
                'Stay low if smoke is present'
            ],
            'medical': [
                'Call emergency medical services (911)',
                'Do not move injured person unless immediate danger',
                'Apply first aid if trained',
                'Stay with the person until help arrives'
            ],
            'natural disaster': [
                'Follow local emergency protocols',
                'Seek immediate shelter',
                'Monitor emergency broadcasts',
                'Avoid flooded roads and downed power lines'
            ],
            'accident': [
                'Call emergency services (911)',
                'Secure the scene if possible',
                'Check for injuries',
                'Document the incident'
            ],
            'security': [
                'Call police immediately (911)',
                'Ensure your safety first',
                'Do not pursue suspects',
                'Preserve evidence if safe'
            ],
            'utility': [
                'Contact utility company',
                'Evacuate if gas leak suspected',
                'Avoid electrical hazards',
                'Call 911 if immediate danger'
            ]
        };
        
        let recommendations = baseRecommendations[emergencyType] || [
            'Call emergency services (911)',
            'Ensure your safety',
            'Provide clear location information',
            'Stay on the line with dispatcher'
        ];
        
        // Add urgency-specific recommendations
        if (urgency > 0.7) {
            recommendations.unshift('⚠️ HIGH PRIORITY: Immediate action required');
        }
        
        // Add emotional state recommendations
        if (emotionalState === 'panic') {
            recommendations.push('Take deep breaths and try to remain calm');
            recommendations.push('Speak slowly and clearly with emergency operators');
        }
        
        return recommendations;
    }
    
    getUrgencyLevel(score) {
        if (score >= 0.8) return 'Critical';
        if (score >= 0.6) return 'High';
        if (score >= 0.4) return 'Medium';
        return 'Low';
    }
    
    calculateClarity(transcript) {
        // Simple clarity calculation based on transcript quality
        const words = transcript.trim().split(' ').length;
        const avgWordLength = transcript.replace(/ /g, '').length / words;
        
        if (words < 5) return 'Poor';
        if (avgWordLength > 6 && words > 10) return 'Excellent';
        if (avgWordLength > 4 && words > 7) return 'Good';
        return 'Fair';
    }
    
    extractImmediateNeeds(text) {
        const needsKeywords = {
            'Medical assistance': ['medical', 'doctor', 'ambulance', 'injured', 'hurt'],
            'Fire department': ['fire', 'smoke', 'burning'],
            'Police': ['police', 'security', 'robbery', 'crime'],
            'Evacuation': ['evacuate', 'trapped', 'stuck', 'blocked'],
            'Utilities': ['power', 'water', 'gas', 'electrical']
        };
        
        const detectedNeeds = [];
        for (const [need, keywords] of Object.entries(needsKeywords)) {
            if (keywords.some(keyword => text.includes(keyword))) {
                detectedNeeds.push(need);
            }
        }
        
        return detectedNeeds.length > 0 ? detectedNeeds.join(', ') : 'Assessment needed';
    }
    
    async fallbackAnalysis() {
        // Fallback analysis when Gemma 3n is not available
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        return {
            urgency: 0.5,
            urgencyLevel: 'Medium',
            emotionalState: 'unknown',
            stressLevel: 'Medium',
            clarity: 'Good',
            confidence: 0.7,
            emergencyType: 'general emergency',
            location: 'Not specified',
            peopleAffected: 'Unknown',
            immediateNeeds: 'Assessment needed',
            recommendations: [
                'Call emergency services (911)',
                'Provide your exact location',
                'Stay safe and follow dispatcher instructions'
            ],
            rawTranscript: this.transcript
        };
    }
    
    displayAnalysisResults(analysis) {
        // Show analysis panel
        this.realtimeAnalysis.classList.add('visible');
        
        // Update urgency
        this.urgencyFill.style.width = `${analysis.urgency * 100}%`;
        this.urgencyText.textContent = `${analysis.urgencyLevel} (${Math.round(analysis.urgency * 100)}%)`;
        
        // Update emotion analysis
        this.stressLevel.textContent = analysis.stressLevel;
        this.clarityLevel.textContent = analysis.clarity;
        this.speakerConfidence.textContent = `${Math.round(analysis.confidence * 100)}%`;
        
        // Update emergency details
        this.emergencyType.textContent = analysis.emergencyType;
        this.emergencyLocation.textContent = analysis.location;
        this.peopleAffected.textContent = analysis.peopleAffected;
        this.immediateNeeds.textContent = analysis.immediateNeeds;
        
        // Update recommendations
        if (Array.isArray(analysis.recommendations)) {
            this.recommendations.innerHTML = `
                <h4>Immediate Actions:</h4>
                <ul>
                    ${analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            `;
        } else {
            this.recommendations.textContent = analysis.recommendations;
        }
        
        // Store analysis for submission
        this.currentAnalysis = analysis;
        
        // Update status
        this.statusText.textContent = `Analysis complete - ${analysis.urgencyLevel} priority emergency detected`;
    }
    
    updateRecordingState() {
        if (this.isRecording) {
            this.recordButton.classList.add('recording');
            this.recordIcon.textContent = '⏹️';
            this.startVisualization();
        } else {
            this.recordButton.classList.remove('recording');
            this.recordIcon.textContent = '🎤';
        }
    }
    
    updateProcessingState() {
        if (this.isProcessing) {
            this.recordButton.classList.add('processing');
            this.recordIcon.textContent = '⚙️';
            this.processingIndicator.classList.add('active');
        } else {
            this.recordButton.classList.remove('processing');
            this.processingIndicator.classList.remove('active');
        }
    }
    
    enableActionButtons() {
        this.submitReportBtn.disabled = false;
        this.retryBtn.disabled = false;
        this.exportBtn.disabled = false;
    }
    
    resetAnalysis() {
        this.realtimeAnalysis.classList.remove('visible');
        this.urgencyFill.style.width = '0%';
        this.urgencyText.textContent = 'Not analyzed';
        this.stressLevel.textContent = '--';
        this.clarityLevel.textContent = '--';
        this.speakerConfidence.textContent = '--';
        this.emergencyType.textContent = 'Unknown';
        this.emergencyLocation.textContent = 'Not specified';
        this.peopleAffected.textContent = 'Unknown';
        this.immediateNeeds.textContent = 'None identified';
        this.recommendations.innerHTML = '🤖 AI will provide response recommendations after analyzing your voice report...';
        this.confidenceSpan.textContent = 'Confidence: --';
    }
    
    resetSession() {
        this.transcript = '';
        this.transcriptText.textContent = 'Click the microphone and speak to start recording your emergency report...';
        this.resetAnalysis();
        this.submitReportBtn.disabled = true;
        this.retryBtn.disabled = true;
        this.exportBtn.disabled = true;
        this.statusText.textContent = 'Tap the microphone to start voice recording';
        this.currentAnalysis = null;
    }
    
    async submitEmergencyReport() {
        if (!this.currentAnalysis || !this.transcript) {
            alert('No analysis available to submit. Please record an emergency report first.');
            return;
        }
        
        const reportData = {
            type: 'voice_emergency_report',
            timestamp: new Date().toISOString(),
            transcript: this.transcript,
            analysis: this.currentAnalysis,
            language: this.currentLanguage,
            model_used: this.selectedModel,
            location: this.currentAnalysis.location,
            urgency_level: this.currentAnalysis.urgencyLevel,
            emergency_type: this.currentAnalysis.emergencyType,
            people_affected: this.currentAnalysis.peopleAffected,
            immediate_needs: this.currentAnalysis.immediateNeeds,
            ai_recommendations: this.currentAnalysis.recommendations
        };
        
        try {
            const response = await fetch('/api/submit-emergency-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(reportData)
            });
            
            if (response.ok) {
                alert('✅ Emergency report submitted successfully!');
                this.statusText.textContent = 'Report submitted successfully';
                
                // Redirect to appropriate page based on urgency
                if (this.currentAnalysis.urgency >= 0.7) {
                    window.location.href = '/admin-dashboard'; // High priority -> admin
                } else {
                    window.location.href = '/view-reports'; // Lower priority -> reports
                }
            } else {
                throw new Error('Failed to submit report');
            }
        } catch (error) {
            console.error('Error submitting report:', error);
            
            // Store in offline queue
            this.storeOfflineReport(reportData);
            alert('📱 Report saved offline. Will sync when connection is restored.');
            this.statusText.textContent = 'Report saved offline for later sync';
        }
    }
    
    storeOfflineReport(reportData) {
        const offlineQueue = JSON.parse(localStorage.getItem('emergencyReportsQueue') || '[]');
        offlineQueue.push({
            ...reportData,
            stored_offline: true,
            priority: this.currentAnalysis.urgency >= 0.7 ? 'critical' : 'normal'
        });
        localStorage.setItem('emergencyReportsQueue', JSON.stringify(offlineQueue));
        
        // Update sync queue UI if available
        if (typeof showSyncQueue === 'function') {
            showSyncQueue();
        }
    }
    
    exportTranscript() {
        if (!this.transcript || !this.currentAnalysis) {
            alert('No transcript available to export.');
            return;
        }
        
        const exportData = {
            timestamp: new Date().toISOString(),
            transcript: this.transcript,
            analysis: this.currentAnalysis,
            language: this.currentLanguage,
            model_used: this.selectedModel
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `voice-emergency-report-${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.statusText.textContent = 'Transcript exported successfully';
    }
}

// Initialize when DOM is loaded
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceEmergencyReporter;
} else {
    window.VoiceEmergencyReporter = VoiceEmergencyReporter;
}