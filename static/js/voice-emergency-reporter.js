/**
 * Voice Emergency Reporter - Gemma 3n Powered Voice Analysis
 * Handles real-time audio processing, transcription, and AI analysis for the emergency portal.
 *
 * @class VoiceEmergencyReporter
 */
class VoiceEmergencyReporter {
    constructor() {
        this.isRecording = false;
        this.isProcessing = false;
        this.transcript = '';
        this.finalizedTranscript = '';
        this.audioContext = null;
        this.analyzer = null;
        this.microphone = null;
        this.dataArray = null;
        this.visualizerBars = [];
        this.currentLanguage = 'en-US';
        this.sensitivity = 0.7;
        this.selectedModel = 'gemma-3n-4b';
        this.autoSubmit = false;
        this.currentAnalysis = null;

        this.initializeElements();
        this.initializeAudioProcessing();
        this.setupEventListeners();
        this.createAudioVisualizer();
        this.initializeGemma3n();
    }

    initializeElements() {
        // Main Controls
        this.recordButton = document.getElementById('recordButton');
        this.recordIcon = document.getElementById('recordIcon');
        this.statusText = document.getElementById('statusText');
        
        // Transcript & Confidence
        this.transcriptContainer = document.getElementById('transcript-container');
        this.transcriptText = document.getElementById('transcriptText');
        this.confidenceSpan = document.getElementById('confidence');
        
        // AI Confidence Meter (New)
        this.aiConfidenceMeter = document.getElementById('aiConfidence');
        this.confidenceScore = document.getElementById('confidenceScore');
        this.confidenceFill = document.getElementById('confidenceFill');

        // Analysis Section
        this.realtimeAnalysis = document.getElementById('realtimeAnalysis');
        this.urgencyFill = document.getElementById('urgencyFill');
        this.urgencyText = document.getElementById('urgencyText');
        this.stressLevel = document.getElementById('stressLevel');
        this.clarityLevel = document.getElementById('clarityLevel');
        this.speakerConfidence = document.getElementById('speakerConfidence');
        this.emergencyType = document.getElementById('emergencyType');
        this.emergencyLocation = document.getElementById('emergencyLocation');
        this.peopleAffected = document.getElementById('peopleAffected');
        this.immediateNeeds = document.getElementById('immediateNeeds');
        this.recommendations = document.getElementById('recommendations');
        
        // Action Buttons
        this.submitReportBtn = document.getElementById('submitReportBtn');
        this.retryBtn = document.getElementById('retryBtn');
        this.exportBtn = document.getElementById('exportBtn');
        
        // Indicators
        this.processingIndicator = document.getElementById('processingIndicator');

        // Settings
        this.languageSelect = document.getElementById('languageSelect');
        this.sensitivitySlider = document.getElementById('sensitivitySlider');
        this.modelSelect = document.getElementById('modelSelect');
        this.autoSubmitToggle = document.getElementById('autoSubmitToggle');
    }

    async initializeGemma3n() {
        this.updateAIStatus('loading');
        try {
            if (window.EdgeAI && typeof window.EdgeAI.loadAudioModel === 'function') {
                await window.EdgeAI.loadAudioModel(this.selectedModel);
                console.log('✅ Gemma 3n audio model loaded:', this.selectedModel);
                this.updateAIStatus('ready');
            } else {
                console.warn('⚠️ EdgeAI or loadAudioModel not available, using fallback.');
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
        if (!aiStatusDot || !aiStatusText) return;

        aiStatusDot.className = 'ai-status-dot'; // Reset classes
        switch (status) {
            case 'ready':
                aiStatusText.textContent = '🧠 Gemma 3n Ready for Voice Analysis';
                break;
            case 'processing':
                aiStatusDot.classList.add('loading');
                aiStatusText.textContent = '🧠 Analyzing with Gemma 3n...';
                break;
            case 'fallback':
                aiStatusDot.classList.add('loading');
                aiStatusText.textContent = '🧠 Using Fallback AI Processing';
                break;
            case 'error':
                aiStatusDot.classList.add('error');
                aiStatusText.textContent = '🧠 AI Processing Unavailable';
                break;
            default:
                aiStatusDot.classList.add('loading');
                aiStatusText.textContent = '🧠 Loading Edge AI models...';
                break;
        }
    }

    initializeAudioProcessing() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.error('Speech recognition not supported in this browser.');
            if(this.statusText) this.statusText.textContent = 'Speech recognition not supported.';
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = this.currentLanguage;

        this.recognition.onstart = () => this.updateRecordingState(true);
        this.recognition.onresult = (event) => this.handleSpeechResult(event);
        this.recognition.onend = () => {
            this.updateRecordingState(false);
            if (this.finalizedTranscript.trim()) {
                this.processWithGemma3n();
            }
        };
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            if(this.statusText) this.statusText.textContent = `Error: ${event.error}`;
            this.updateRecordingState(false);
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
            
            this.dataArray = new Uint8Array(this.analyzer.frequencyBinCount);
            this.startVisualization();
        } catch (error) {
            console.error('Error setting up audio visualizer:', error);
            if(this.statusText) this.statusText.textContent = "Microphone access denied.";
        }
    }

    createAudioVisualizer() {
        const visualizer = document.getElementById('audioVisualizer');
        if (!visualizer) return;
        visualizer.innerHTML = '';
        this.visualizerBars = [];
        for (let i = 0; i < 20; i++) {
            const bar = document.createElement('div');
            bar.className = 'audio-bar';
            bar.style.height = '4px';
            visualizer.appendChild(bar);
            this.visualizerBars.push(bar);
        }
    }

    startVisualization() {
        if (!this.analyzer) return;
        const animate = () => {
            if (!this.isRecording) {
                this.visualizerBars.forEach(bar => bar.style.height = '4px');
                return;
            }
            this.analyzer.getByteFrequencyData(this.dataArray);
            this.visualizerBars.forEach((bar, index) => {
                const value = this.dataArray[index * 6] || 0; // Use a stride to get varied values
                const height = Math.max(2, (value / 255) * 58); // Max height of 60px
                bar.style.height = `${height}px`;
            });
            requestAnimationFrame(animate);
        };
        animate();
    }

    setupEventListeners() {
        this.recordButton?.addEventListener('click', () => this.isRecording ? this.stopRecording() : this.startRecording());
        this.languageSelect?.addEventListener('change', (e) => {
            this.currentLanguage = e.target.value;
            if (this.recognition) this.recognition.lang = this.currentLanguage;
        });
        this.sensitivitySlider?.addEventListener('input', (e) => this.sensitivity = parseFloat(e.target.value));
        this.modelSelect?.addEventListener('change', (e) => {
            this.selectedModel = e.target.value;
            this.initializeGemma3n();
        });
        this.autoSubmitToggle?.addEventListener('change', (e) => this.autoSubmit = e.target.checked);
        this.submitReportBtn?.addEventListener('click', () => this.submitEmergencyReport());
        this.retryBtn?.addEventListener('click', () => this.resetSession());
        this.exportBtn?.addEventListener('click', () => this.exportTranscript());
    }

    async startRecording() {
        if (!this.recognition) return;
        this.resetSession();
        if (!this.audioContext) await this.setupAudioVisualizer();
        this.recognition.start();
    }

    stopRecording() {
        if (this.recognition) this.recognition.stop();
    }

    updateRecordingState(isRecording) {
        this.isRecording = isRecording;
        if (isRecording) {
            this.recordButton?.classList.add('recording');
            if (this.recordIcon) this.recordIcon.textContent = '⏹️';
            if (this.statusText) this.statusText.textContent = 'Listening... Speak clearly.';
            this.startVisualization();
        } else {
            this.recordButton?.classList.remove('recording');
            if (this.recordIcon) this.recordIcon.textContent = '🎤';
            if (this.statusText) this.statusText.textContent = 'Recording stopped. Processing...';
        }
    }

    handleSpeechResult(event) {
        let interimTranscript = '';
        this.finalizedTranscript = this.transcript; // Keep track of last final transcript

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcriptSegment = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                this.finalizedTranscript += transcriptSegment + ' ';
            } else {
                interimTranscript += transcriptSegment;
            }
        }

        // Update UI with both final and interim results for live feedback
        this.updateTranscriptDisplay(this.finalizedTranscript, interimTranscript);
    }
    
    updateTranscriptDisplay(finalText, interimText) {
        if (!this.transcriptText) return;
        
        let liveIndicatorHTML = '';
        if (interimText) {
            liveIndicatorHTML = `
                <div class="transcription-live">
                    ${interimText}
                    <div class="typing-indicator"></div>
                    <div class="typing-indicator"></div>
                    <div class="typing-indicator"></div>
                </div>
            `;
        }
        
        this.transcriptText.innerHTML = `${finalText}${liveIndicatorHTML}`;
    }

    async processWithGemma3n() {
        this.transcript = this.finalizedTranscript.trim();
        if (!this.transcript) {
            this.statusText.textContent = 'No speech detected. Please try again.';
            this.resetSession();
            return;
        }
        
        this.isProcessing = true;
        this.updateProcessingState();
        this.updateAIStatus('processing');
        
        try {
            const analysisResult = (window.EdgeAI && window.EdgeAI.isReady)
                ? await this.simulateGemma3nAnalysis(this.transcript)
                : await this.fallbackAnalysis();
            
            this.displayAnalysisResults(analysisResult);
            this.enableActionButtons();
            
            if (this.autoSubmit && analysisResult.urgency >= 0.7) {
                setTimeout(() => this.submitEmergencyReport(), 2000);
            }
        } catch (error) {
            console.error('Error processing with AI:', error);
            if(this.statusText) this.statusText.textContent = 'Error processing speech.';
        } finally {
            this.isProcessing = false;
            this.updateProcessingState();
            this.updateAIStatus('ready');
        }
    }

    displayAnalysisResults(analysis) {
        this.realtimeAnalysis.style.display = 'block';
        this.currentAnalysis = analysis;
        
        // Populate AI Confidence Meter
        if (this.aiConfidenceMeter) {
            this.aiConfidenceMeter.style.display = 'block';
            this.confidenceScore.textContent = `${Math.round(analysis.confidence * 100)}%`;
            this.confidenceFill.style.width = `${analysis.confidence * 100}%`;
        }

        // Update UI elements
        this.urgencyFill.style.width = `${analysis.urgency * 100}%`;
        this.urgencyText.textContent = `${analysis.urgencyLevel} (${Math.round(analysis.urgency * 100)}%)`;
        this.stressLevel.textContent = analysis.stressLevel;
        this.clarityLevel.textContent = analysis.clarity;
        this.speakerConfidence.textContent = `${Math.round(analysis.confidence * 100)}%`;
        this.emergencyType.textContent = analysis.emergencyType;
        this.emergencyLocation.textContent = analysis.location;
        this.peopleAffected.textContent = analysis.peopleAffected;
        this.immediateNeeds.textContent = analysis.immediateNeeds;
        this.recommendations.innerHTML = `<h4>Immediate Actions:</h4><ul>${analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}</ul>`;
        this.statusText.textContent = `Analysis complete: ${analysis.urgencyLevel} priority detected.`;
    }

    updateProcessingState() {
        if (this.isProcessing) {
            this.recordButton?.classList.add('processing');
            if (this.recordIcon) this.recordIcon.textContent = '⚙️';
            this.processingIndicator?.classList.add('active');
        } else {
            this.recordButton?.classList.remove('processing');
            this.processingIndicator?.classList.remove('active');
        }
    }

    enableActionButtons() {
        if(this.submitReportBtn) this.submitReportBtn.disabled = false;
        if(this.retryBtn) this.retryBtn.disabled = false;
        if(this.exportBtn) this.exportBtn.disabled = false;
    }

    resetSession() {
        this.transcript = '';
        this.finalizedTranscript = '';
        if(this.transcriptText) this.transcriptText.innerHTML = 'Click the microphone and speak...';
        if(this.realtimeAnalysis) this.realtimeAnalysis.style.display = 'none';
        if(this.aiConfidenceMeter) this.aiConfidenceMeter.style.display = 'none';
        if(this.submitReportBtn) this.submitReportBtn.disabled = true;
        if(this.retryBtn) this.retryBtn.disabled = true;
        if(this.exportBtn) this.exportBtn.disabled = true;
        if(this.statusText) this.statusText.textContent = 'Tap the microphone to start';
        this.currentAnalysis = null;
    }

    async submitEmergencyReport() {
        if (!this.currentAnalysis) {
            alert('No analysis available.');
            return;
        }
        // ... (rest of your submit logic)
        alert('Report submitted successfully! (Simulated)');
    }
    
    // ... (Your other methods like simulateGemma3nAnalysis, fallbackAnalysis, exportTranscript, etc. would go here)
    // The placeholder functions from the original file are sufficient.
    async simulateGemma3nAnalysis(transcript) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        const urgencyScore = Math.random();
        return {
            urgency: urgencyScore,
            urgencyLevel: this.getUrgencyLevel(urgencyScore),
            emotionalState: 'Stressed',
            stressLevel: 'High',
            clarity: 'Good',
            confidence: 0.85 + Math.random() * 0.1,
            emergencyType: 'Medical',
            location: 'Near Main Street',
            peopleAffected: '1 person',
            immediateNeeds: 'Ambulance',
            recommendations: ['Call 911', 'Do not move the patient'],
            rawTranscript: transcript
        };
    }
    
    getUrgencyLevel(score) {
        if (score >= 0.8) return 'Critical';
        if (score >= 0.6) return 'High';
        if (score >= 0.4) return 'Medium';
        return 'Low';
    }
    
    exportTranscript() {
        if (!this.transcript) {
            alert('No transcript to export.');
            return;
        }
        const dataStr = JSON.stringify({ transcript: this.transcript, analysis: this.currentAnalysis }, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'emergency-report.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}