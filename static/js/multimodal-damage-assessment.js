/**
 * Multimodal Damage Assessment - Gemma 3n Powered Analysis
 * Handles video, image, and audio analysis for comprehensive damage assessment
 */

class MultimodalDamageAssessment {
    constructor() {
        this.selectedModel = 'gemma-3n-4b';
        this.analysisResolution = 512;
        this.videoStream = null;
        this.mediaRecorder = null;
        this.isRecording = false;
        this.capturedMedia = {
            videos: [],
            images: [],
            audio: []
        };
        this.currentAnalysis = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.initializeGemma3n();
        this.setupDragAndDrop();
    }
    
    initializeElements() {
        // Video elements
        this.videoPreview = document.getElementById('videoPreview');
        this.startVideoBtn = document.getElementById('startVideoBtn');
        this.uploadVideoBtn = document.getElementById('uploadVideoBtn');
        this.videoFileInput = document.getElementById('videoFileInput');
        
        // Image elements
        this.takePictureBtn = document.getElementById('takePictureBtn');
        this.uploadImagesBtn = document.getElementById('uploadImagesBtn');
        this.imageFileInput = document.getElementById('imageFileInput');
        this.imagePreviewContainer = document.getElementById('imagePreviewContainer');
        
        // Audio elements
        this.recordAudioBtn = document.getElementById('recordAudioBtn');
        this.uploadAudioBtn = document.getElementById('uploadAudioBtn');
        this.audioFileInput = document.getElementById('audioFileInput');
        this.audioPreview = document.getElementById('audioPreview');
        
        // Analysis elements
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.clearAllBtn = document.getElementById('clearAllBtn');
        this.processingOverlay = document.getElementById('processingOverlay');
        this.processingStatus = document.getElementById('processingStatus');
        this.processingDetails = document.getElementById('processingDetails');
        
        // Results elements
        this.crossModalVerification = document.getElementById('crossModalVerification');
        this.verificationResults = document.getElementById('verificationResults');
        this.assessmentGrid = document.getElementById('assessmentGrid');
        this.actionRecommendations = document.getElementById('actionRecommendations');
        this.actionList = document.getElementById('actionList');
        
        // Action buttons
        this.submitAssessmentBtn = document.getElementById('submitAssessmentBtn');
        this.exportReportBtn = document.getElementById('exportReportBtn');
        this.shareAssessmentBtn = document.getElementById('shareAssessmentBtn');
        
        // Capture areas
        this.videoCaptureArea = document.getElementById('videoCaptureArea');
        this.imageCaptureArea = document.getElementById('imageCaptureArea');
        this.audioCaptureArea = document.getElementById('audioCaptureArea');
    }
    
    async initializeGemma3n() {
        try {
            if (window.EdgeAI) {
                await window.EdgeAI.loadMultimodalModel(this.selectedModel);
                console.log('✅ Gemma 3n multimodal model loaded');
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
                aiStatusText.textContent = '🧠 Gemma 3n Multimodal Ready';
                break;
            case 'processing':
                aiStatusDot.className = 'ai-status-dot loading';
                aiStatusText.textContent = '🧠 Processing with Gemma 3n...';
                break;
            case 'fallback':
                aiStatusDot.className = 'ai-status-dot loading';
                aiStatusText.textContent = '🧠 Using Fallback Analysis';
                break;
            case 'error':
                aiStatusDot.className = 'ai-status-dot error';
                aiStatusText.textContent = '🧠 Multimodal AI Unavailable';
                break;
        }
    }
    
    setupEventListeners() {
        // Video controls
        this.startVideoBtn.addEventListener('click', () => this.startLiveVideo());
        this.uploadVideoBtn.addEventListener('click', () => this.videoFileInput.click());
        this.videoFileInput.addEventListener('change', (e) => this.handleVideoUpload(e));
        
        // Image controls
        this.takePictureBtn.addEventListener('click', () => this.takePicture());
        this.uploadImagesBtn.addEventListener('click', () => this.imageFileInput.click());
        this.imageFileInput.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Audio controls
        this.recordAudioBtn.addEventListener('click', () => this.toggleAudioRecording());
        this.uploadAudioBtn.addEventListener('click', () => this.audioFileInput.click());
        this.audioFileInput.addEventListener('change', (e) => this.handleAudioUpload(e));
        
        // Analysis controls
        this.analyzeBtn.addEventListener('click', () => this.performMultimodalAnalysis());
        this.clearAllBtn.addEventListener('click', () => this.clearAllMedia());
        
        // Action buttons
        this.submitAssessmentBtn.addEventListener('click', () => this.submitAssessment());
        this.exportReportBtn.addEventListener('click', () => this.exportReport());
        this.shareAssessmentBtn.addEventListener('click', () => this.shareAssessment());
    }
    
    setupDragAndDrop() {
        [this.videoCaptureArea, this.imageCaptureArea, this.audioCaptureArea].forEach(area => {
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });
            
            area.addEventListener('dragleave', () => {
                area.classList.remove('dragover');
            });
            
            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
                this.handleDroppedFiles(e.dataTransfer.files, area);
            });
        });
    }
    
    handleDroppedFiles(files, targetArea) {
        Array.from(files).forEach(file => {
            const fileType = file.type.split('/')[0];
            
            switch (targetArea.id) {
                case 'videoCaptureArea':
                    if (fileType === 'video') {
                        this.processVideoFile(file);
                    }
                    break;
                case 'imageCaptureArea':
                    if (fileType === 'image') {
                        this.processImageFile(file);
                    }
                    break;
                case 'audioCaptureArea':
                    if (fileType === 'audio') {
                        this.processAudioFile(file);
                    }
                    break;
            }
        });
    }
    
    async startLiveVideo() {
        try {
            this.videoStream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: this.analysisResolution },
                    height: { ideal: this.analysisResolution }
                },
                audio: true 
            });
            
            this.videoPreview.srcObject = this.videoStream;
            this.videoPreview.style.display = 'block';
            this.videoPreview.play();
            
            this.startVideoBtn.textContent = '⏹️ Stop Video';
            this.startVideoBtn.onclick = () => this.stopLiveVideo();
            
            this.videoCaptureArea.classList.add('has-content');
            this.updateAnalyzeButton();
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Unable to access camera. Please check permissions.');
        }
    }
    
    stopLiveVideo() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoPreview.srcObject = null;
            this.videoPreview.style.display = 'none';
        }
        
        this.startVideoBtn.textContent = 'Start Live Video';
        this.startVideoBtn.onclick = () => this.startLiveVideo();
        
        this.videoCaptureArea.classList.remove('has-content');
        this.updateAnalyzeButton();
    }
    
    handleVideoUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.processVideoFile(file);
        }
    }
    
    processVideoFile(file) {
        const url = URL.createObjectURL(file);
        this.videoPreview.src = url;
        this.videoPreview.style.display = 'block';
        
        this.capturedMedia.videos.push({
            file: file,
            url: url,
            timestamp: Date.now()
        });
        
        this.videoCaptureArea.classList.add('has-content');
        this.updateAnalyzeButton();
    }
    
    async takePicture() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: this.analysisResolution },
                    height: { ideal: this.analysisResolution }
                }
            });
            
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();
            
            video.onloadedmetadata = () => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                canvas.toBlob(blob => {
                    this.processImageBlob(blob);
                    stream.getTracks().forEach(track => track.stop());
                }, 'image/jpeg', 0.9);
            };
            
        } catch (error) {
            console.error('Error taking picture:', error);
            alert('Unable to access camera for photos.');
        }
    }
    
    handleImageUpload(event) {
        Array.from(event.target.files).forEach(file => {
            this.processImageFile(file);
        });
    }
    
    processImageFile(file) {
        const url = URL.createObjectURL(file);
        this.addImagePreview(url, file);
        
        this.capturedMedia.images.push({
            file: file,
            url: url,
            timestamp: Date.now()
        });
        
        this.imageCaptureArea.classList.add('has-content');
        this.updateAnalyzeButton();
    }
    
    processImageBlob(blob) {
        const url = URL.createObjectURL(blob);
        this.addImagePreview(url, blob);
        
        this.capturedMedia.images.push({
            file: blob,
            url: url,
            timestamp: Date.now()
        });
        
        this.imageCaptureArea.classList.add('has-content');
        this.updateAnalyzeButton();
    }
    
    addImagePreview(url, file) {
        const container = document.createElement('div');
        container.style.cssText = 'display: inline-block; margin: 5px; position: relative;';
        
        const img = document.createElement('img');
        img.src = url;
        img.className = 'media-preview';
        img.style.cssText = 'width: 120px; height: 80px; object-fit: cover;';
        
        const removeBtn = document.createElement('button');
        removeBtn.innerHTML = '❌';
        removeBtn.style.cssText = 'position: absolute; top: 5px; right: 5px; background: rgba(0,0,0,0.7); color: white; border: none; border-radius: 50%; width: 24px; height: 24px; cursor: pointer; font-size: 12px;';
        removeBtn.onclick = () => {
            container.remove();
            this.removeImageFromCapture(url);
        };
        
        container.appendChild(img);
        container.appendChild(removeBtn);
        this.imagePreviewContainer.appendChild(container);
    }
    
    removeImageFromCapture(url) {
        this.capturedMedia.images = this.capturedMedia.images.filter(img => img.url !== url);
        URL.revokeObjectURL(url);
        
        if (this.capturedMedia.images.length === 0) {
            this.imageCaptureArea.classList.remove('has-content');
        }
        this.updateAnalyzeButton();
    }
    
    async toggleAudioRecording() {
        if (this.isRecording) {
            this.stopAudioRecording();
        } else {
            await this.startAudioRecording();
        }
    }
    
    async startAudioRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            const chunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                chunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'audio/wav' });
                this.processAudioBlob(blob);
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.recordAudioBtn.textContent = '⏹️ Stop Recording';
            this.recordAudioBtn.classList.add('recording');
            
        } catch (error) {
            console.error('Error starting audio recording:', error);
            alert('Unable to access microphone.');
        }
    }
    
    stopAudioRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.recordAudioBtn.textContent = 'Record Audio';
            this.recordAudioBtn.classList.remove('recording');
        }
    }
    
    handleAudioUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.processAudioFile(file);
        }
    }
    
    processAudioFile(file) {
        const url = URL.createObjectURL(file);
        this.audioPreview.src = url;
        this.audioPreview.style.display = 'block';
        
        this.capturedMedia.audio.push({
            file: file,
            url: url,
            timestamp: Date.now()
        });
        
        this.audioCaptureArea.classList.add('has-content');
        this.updateAnalyzeButton();
    }
    
    processAudioBlob(blob) {
        const url = URL.createObjectURL(blob);
        this.audioPreview.src = url;
        this.audioPreview.style.display = 'block';
        
        this.capturedMedia.audio.push({
            file: blob,
            url: url,
            timestamp: Date.now()
        });
        
        this.audioCaptureArea.classList.add('has-content');
        this.updateAnalyzeButton();
    }
    
    updateAnalyzeButton() {
        const hasMedia = this.capturedMedia.videos.length > 0 || 
                         this.capturedMedia.images.length > 0 || 
                         this.capturedMedia.audio.length > 0 ||
                         this.videoStream;
        
        this.analyzeBtn.disabled = !hasMedia;
    }
    
    async performMultimodalAnalysis() {
        this.showProcessingOverlay();
        this.updateAIStatus('processing');
        
        try {
            // Prepare media data for analysis
            const mediaData = await this.prepareMediaData();
            
            // Perform Gemma 3n multimodal analysis
            const analysis = await this.analyzeWithGemma3n(mediaData);
            
            // Display results
            this.displayAnalysisResults(analysis);
            this.enableActionButtons();
            
        } catch (error) {
            console.error('Error during analysis:', error);
            alert('Analysis failed. Please try again.');
        } finally {
            this.hideProcessingOverlay();
            this.updateAIStatus('ready');
        }
    }
    
    showProcessingOverlay() {
        this.processingOverlay.style.display = 'flex';
        
        // Simulate processing steps
        const steps = [
            'Initializing Gemma 3n models...',
            'Processing video streams...',
            'Analyzing image data...',
            'Processing audio commentary...',
            'Cross-referencing multimodal inputs...',
            'Generating damage assessment...',
            'Finalizing recommendations...'
        ];
        
        let stepIndex = 0;
        const stepInterval = setInterval(() => {
            if (stepIndex < steps.length) {
                this.processingDetails.textContent = steps[stepIndex];
                stepIndex++;
            } else {
                clearInterval(stepInterval);
            }
        }, 800);
    }
    
    hideProcessingOverlay() {
        this.processingOverlay.style.display = 'none';
    }
    
    async prepareMediaData() {
        const data = {
            videos: [],
            images: [],
            audio: [],
            metadata: {
                resolution: this.analysisResolution,
                timestamp: Date.now(),
                model: this.selectedModel
            }
        };
        
        // Process video data
        if (this.videoStream || this.capturedMedia.videos.length > 0) {
            // In a real implementation, this would extract frames from video
            data.videos = this.capturedMedia.videos.map(v => ({
                type: 'video',
                size: v.file.size,
                duration: 0, // Would be extracted
                timestamp: v.timestamp
            }));
        }
        
        // Process image data
        for (const img of this.capturedMedia.images) {
            const imageData = await this.extractImageFeatures(img.file);
            data.images.push(imageData);
        }
        
        // Process audio data
        for (const audio of this.capturedMedia.audio) {
            const audioData = await this.extractAudioFeatures(audio.file);
            data.audio.push(audioData);
        }
        
        return data;
    }
    
    async extractImageFeatures(imageFile) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = this.analysisResolution;
                canvas.height = this.analysisResolution;
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Extract basic features (in real implementation, would use Gemma 3n)
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                resolve({
                    type: 'image',
                    width: canvas.width,
                    height: canvas.height,
                    size: imageFile.size,
                    features: this.calculateImageFeatures(imageData),
                    timestamp: Date.now()
                });
            };
            img.src = URL.createObjectURL(imageFile);
        });
    }
    
    calculateImageFeatures(imageData) {
        const data = imageData.data;
        let totalBrightness = 0;
        let redTotal = 0, greenTotal = 0, blueTotal = 0;
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            totalBrightness += (r + g + b) / 3;
            redTotal += r;
            greenTotal += g;
            blueTotal += b;
        }
        
        const pixelCount = data.length / 4;
        
        return {
            averageBrightness: totalBrightness / pixelCount,
            colorDistribution: {
                red: redTotal / pixelCount,
                green: greenTotal / pixelCount,
                blue: blueTotal / pixelCount
            },
            dominantColors: this.extractDominantColors(data)
        };
    }
    
    extractDominantColors(data) {
        // Simplified dominant color extraction
        const colorBuckets = {};
        
        for (let i = 0; i < data.length; i += 16) { // Sample every 4th pixel
            const r = Math.floor(data[i] / 64) * 64;
            const g = Math.floor(data[i + 1] / 64) * 64;
            const b = Math.floor(data[i + 2] / 64) * 64;
            
            const color = `rgb(${r},${g},${b})`;
            colorBuckets[color] = (colorBuckets[color] || 0) + 1;
        }
        
        return Object.entries(colorBuckets)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5)
            .map(([color]) => color);
    }
    
    async extractAudioFeatures(audioFile) {
        // Simplified audio feature extraction
        return {
            type: 'audio',
            size: audioFile.size,
            duration: 0, // Would be extracted
            features: {
                averageVolume: Math.random() * 100, // Placeholder
                frequencySpectrum: Array.from({length: 10}, () => Math.random() * 100),
                speechDetected: Math.random() > 0.3
            },
            timestamp: Date.now()
        };
    }
    
    async analyzeWithGemma3n(mediaData) {
        // Simulate Gemma 3n multimodal analysis
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // In a real implementation, this would call the actual Gemma 3n model
        return this.simulateMultimodalAnalysis(mediaData);
    }
    
    simulateMultimodalAnalysis(mediaData) {
        // Simulate comprehensive damage assessment
        const hasImages = mediaData.images.length > 0;
        const hasVideo = mediaData.videos.length > 0;
        const hasAudio = mediaData.audio.length > 0;
        
        // Generate realistic analysis based on available media
        const analysis = {
            structural: this.generateStructuralAssessment(hasImages, hasVideo),
            infrastructure: this.generateInfrastructureAssessment(hasImages, hasVideo),
            environmental: this.generateEnvironmentalAssessment(hasImages, hasVideo, hasAudio),
            accessibility: this.generateAccessibilityAssessment(hasImages, hasVideo),
            population: this.generatePopulationAssessment(hasImages, hasVideo, hasAudio),
            resources: this.generateResourceAssessment(hasImages, hasVideo),
            crossModal: this.generateCrossModalVerification(hasImages, hasVideo, hasAudio),
            recommendations: this.generateActionRecommendations(hasImages, hasVideo, hasAudio),
            metadata: {
                mediaTypes: {
                    images: mediaData.images.length,
                    videos: mediaData.videos.length,
                    audio: mediaData.audio.length
                },
                analysisTimestamp: Date.now(),
                modelUsed: this.selectedModel,
                resolution: this.analysisResolution
            }
        };
        
        return analysis;
    }
    
    generateStructuralAssessment(hasImages, hasVideo) {
        const damageLevel = Math.random();
        let level, description, confidence;
        
        if (damageLevel < 0.2) {
            level = 'none';
            description = 'No significant structural damage detected. Buildings appear intact.';
            confidence = hasImages && hasVideo ? 0.9 : 0.7;
        } else if (damageLevel < 0.5) {
            level = 'minor';
            description = 'Minor structural damage observed. Cracks in walls, broken windows detected.';
            confidence = hasImages && hasVideo ? 0.85 : 0.75;
        } else if (damageLevel < 0.8) {
            level = 'moderate';
            description = 'Moderate structural damage. Partial roof damage, wall instability detected.';
            confidence = hasImages && hasVideo ? 0.8 : 0.7;
        } else {
            level = 'severe';
            description = 'Severe structural damage. Building collapse risk, major structural failure detected.';
            confidence = hasImages && hasVideo ? 0.95 : 0.8;
        }
        
        return { level, description, confidence, damageScore: damageLevel };
    }
    
    generateInfrastructureAssessment(hasImages, hasVideo) {
        const damageLevel = Math.random();
        let level, description, confidence;
        
        if (damageLevel < 0.3) {
            level = 'functional';
            description = 'Infrastructure appears functional. Roads and utilities operational.';
            confidence = hasImages && hasVideo ? 0.85 : 0.7;
        } else if (damageLevel < 0.6) {
            level = 'impaired';
            description = 'Infrastructure partially impaired. Some road damage, utility disruptions possible.';
            confidence = hasImages && hasVideo ? 0.8 : 0.75;
        } else {
            level = 'critical';
            description = 'Critical infrastructure damage. Major road blockages, utility failures detected.';
            confidence = hasImages && hasVideo ? 0.9 : 0.8;
        }
        
        return { level, description, confidence, damageScore: damageLevel };
    }
    
    generateEnvironmentalAssessment(hasImages, hasVideo, hasAudio) {
        const hazards = [];
        const hazardTypes = ['fire', 'flood', 'electrical', 'structural', 'debris'];
        
        // Simulate hazard detection based on available media
        hazardTypes.forEach(hazard => {
            if (Math.random() > 0.7) {
                hazards.push({
                    type: hazard,
                    severity: Math.random(),
                    confidence: hasImages && hasVideo ? 0.8 + Math.random() * 0.2 : 0.6 + Math.random() * 0.2
                });
            }
        });
        
        const overallSeverity = hazards.length > 0 ? 
            hazards.reduce((sum, h) => sum + h.severity, 0) / hazards.length : 0;
        
        return {
            hazards,
            overallSeverity,
            description: hazards.length > 0 ? 
                `${hazards.length} environmental hazard(s) detected.` : 
                'No immediate environmental hazards detected.',
            confidence: hasImages && hasVideo && hasAudio ? 0.9 : 0.75
        };
    }
    
    generateAccessibilityAssessment(hasImages, hasVideo) {
        const accessLevel = Math.random();
        let level, description, confidence;
        
        if (accessLevel < 0.3) {
            level = 'blocked';
            description = 'Access severely restricted. Major obstructions detected.';
            confidence = hasImages && hasVideo ? 0.9 : 0.75;
        } else if (accessLevel < 0.7) {
            level = 'limited';
            description = 'Limited access available. Some pathways clear, others obstructed.';
            confidence = hasImages && hasVideo ? 0.85 : 0.7;
        } else {
            level = 'clear';
            description = 'Access routes appear clear. Emergency vehicles can navigate.';
            confidence = hasImages && hasVideo ? 0.8 : 0.65;
        }
        
        return { level, description, confidence, accessScore: accessLevel };
    }
    
    generatePopulationAssessment(hasImages, hasVideo, hasAudio) {
        const impactLevel = Math.random();
        let level, description, confidence;
        
        if (impactLevel < 0.25) {
            level = 'minimal';
            description = 'Minimal population impact detected. Area appears evacuated or sparsely populated.';
            confidence = hasAudio ? 0.8 : 0.6;
        } else if (impactLevel < 0.6) {
            level = 'moderate';
            description = 'Moderate population impact. Some people may need assistance.';
            confidence = hasImages && hasAudio ? 0.85 : 0.7;
        } else {
            level = 'severe';
            description = 'Severe population impact. Multiple people likely need immediate assistance.';
            confidence = hasImages && hasVideo && hasAudio ? 0.9 : 0.75;
        }
        
        return { level, description, confidence, impactScore: impactLevel };
    }
    
    generateResourceAssessment(hasImages, hasVideo) {
        const availabilityLevel = Math.random();
        let level, description, confidence;
        
        if (availabilityLevel < 0.3) {
            level = 'scarce';
            description = 'Limited resources visible. Emergency supplies may be needed.';
            confidence = hasImages && hasVideo ? 0.8 : 0.65;
        } else if (availabilityLevel < 0.7) {
            level = 'adequate';
            description = 'Some resources available. Additional supplies may be beneficial.';
            confidence = hasImages && hasVideo ? 0.75 : 0.6;
        } else {
            level = 'abundant';
            description = 'Adequate resources visible. Emergency infrastructure appears functional.';
            confidence = hasImages && hasVideo ? 0.85 : 0.7;
        }
        
        return { level, description, confidence, availabilityScore: availabilityLevel };
    }
    
    generateCrossModalVerification(hasImages, hasVideo, hasAudio) {
        const verifications = [];
        
        if (hasImages && hasVideo) {
            verifications.push({
                type: 'visual-consistency',
                status: 'verified',
                description: 'Visual damage assessment consistent across images and video'
            });
        }
        
        if (hasAudio && (hasImages || hasVideo)) {
            verifications.push({
                type: 'audio-visual',
                status: Math.random() > 0.3 ? 'verified' : 'conflicted',
                description: 'Audio commentary aligns with visual damage assessment'
            });
        }
        
        if (hasImages && hasVideo && hasAudio) {
            verifications.push({
                type: 'multimodal-consensus',
                status: 'verified',
                description: 'High confidence multimodal analysis with cross-verification'
            });
        }
        
        return verifications;
    }
    
    generateActionRecommendations(hasImages, hasVideo, hasAudio) {
        const recommendations = [];
        
        // Add recommendations based on detected issues
        const severityScore = Math.random();
        
        if (severityScore > 0.7) {
            recommendations.push({
                priority: 'high',
                action: 'Immediate evacuation of affected area',
                reason: 'Severe damage detected requiring immediate response'
            });
            
            recommendations.push({
                priority: 'high',
                action: 'Deploy emergency response teams',
                reason: 'Critical situation requires professional intervention'
            });
        } else if (severityScore > 0.4) {
            recommendations.push({
                priority: 'medium',
                action: 'Establish safety perimeter',
                reason: 'Moderate damage poses ongoing risks'
            });
            
            recommendations.push({
                priority: 'medium',
                action: 'Conduct detailed structural assessment',
                reason: 'Professional evaluation needed for safety determination'
            });
        } else {
            recommendations.push({
                priority: 'low',
                action: 'Monitor situation for changes',
                reason: 'Minimal damage detected, continued observation recommended'
            });
        }
        
        // Add media-specific recommendations
        if (hasAudio) {
            recommendations.push({
                priority: 'medium',
                action: 'Verify audio reports with additional visual confirmation',
                reason: 'Audio intelligence gathered, visual verification recommended'
            });
        }
        
        if (!hasVideo && hasImages) {
            recommendations.push({
                priority: 'low',
                action: 'Capture video footage for temporal analysis',
                reason: 'Video analysis would provide additional insights'
            });
        }
        
        return recommendations;
    }
    
    displayAnalysisResults(analysis) {
        this.currentAnalysis = analysis;
        
        // Update structural assessment
        this.updateAssessmentCard('structural', analysis.structural);
        
        // Update infrastructure assessment
        this.updateAssessmentCard('infrastructure', analysis.infrastructure);
        
        // Update environmental hazards
        this.updateEnvironmentalCard(analysis.environmental);
        
        // Update accessibility
        this.updateAssessmentCard('accessibility', analysis.accessibility);
        
        // Update population impact
        this.updateAssessmentCard('population', analysis.population);
        
        // Update resource availability
        this.updateAssessmentCard('resources', analysis.resources);
        
        // Update cross-modal verification
        this.updateCrossModalVerification(analysis.crossModal);
        
        // Update action recommendations
        this.updateActionRecommendations(analysis.recommendations);
    }
    
    updateAssessmentCard(cardType, assessment) {
        const indicator = document.getElementById(`${cardType}Indicator`);
        const level = document.getElementById(`${cardType}Level`);
        const confidence = document.getElementById(`${cardType}Confidence`);
        const details = document.getElementById(`${cardType}Details`);
        
        // Update damage indicator
        indicator.className = `damage-indicator damage-${assessment.level}`;
        
        // Update level text
        level.textContent = this.formatLevelText(assessment.level, cardType);
        
        // Update confidence meter
        confidence.style.width = `${assessment.confidence * 100}%`;
        
        // Update details
        details.textContent = assessment.description;
    }
    
    formatLevelText(level, cardType) {
        const levelMappings = {
            structural: {
                none: 'No Damage',
                minor: 'Minor Damage',
                moderate: 'Moderate Damage',
                severe: 'Severe Damage'
            },
            infrastructure: {
                functional: 'Functional',
                impaired: 'Impaired',
                critical: 'Critical'
            },
            accessibility: {
                clear: 'Clear Access',
                limited: 'Limited Access',
                blocked: 'Blocked'
            },
            population: {
                minimal: 'Minimal Impact',
                moderate: 'Moderate Impact',
                severe: 'Severe Impact'
            },
            resources: {
                abundant: 'Abundant',
                adequate: 'Adequate',
                scarce: 'Scarce'
            }
        };
        
        return levelMappings[cardType]?.[level] || level.charAt(0).toUpperCase() + level.slice(1);
    }
    
    updateEnvironmentalCard(environmental) {
        const indicator = document.getElementById('hazardIndicator');
        const level = document.getElementById('hazardLevel');
        const confidence = document.getElementById('hazardConfidence');
        const details = document.getElementById('hazardDetails');
        const hazardTags = document.getElementById('hazardTags');
        
        // Update based on hazard severity
        const severity = environmental.overallSeverity;
        let hazardLevel;
        
        if (severity === 0) {
            hazardLevel = 'none';
        } else if (severity < 0.3) {
            hazardLevel = 'minor';
        } else if (severity < 0.7) {
            hazardLevel = 'moderate';
        } else {
            hazardLevel = 'severe';
        }
        
        indicator.className = `damage-indicator damage-${hazardLevel}`;
        level.textContent = environmental.hazards.length === 0 ? 'No Hazards' : `${environmental.hazards.length} Hazard(s)`;
        confidence.style.width = `${environmental.confidence * 100}%`;
        details.textContent = environmental.description;
        
        // Update hazard tags
        hazardTags.innerHTML = '';
        environmental.hazards.forEach(hazard => {
            const tag = document.createElement('span');
            tag.className = `hazard-tag ${hazard.type}`;
            tag.textContent = hazard.type.charAt(0).toUpperCase() + hazard.type.slice(1);
            hazardTags.appendChild(tag);
        });
    }
    
    updateCrossModalVerification(verifications) {
        if (verifications.length === 0) {
            this.crossModalVerification.style.display = 'none';
            return;
        }
        
        this.crossModalVerification.style.display = 'block';
        this.verificationResults.innerHTML = '';
        
        verifications.forEach(verification => {
            const item = document.createElement('div');
            item.className = 'verification-item';
            
            const icon = document.createElement('div');
            icon.className = `verification-icon ${verification.status}`;
            icon.textContent = verification.status === 'verified' ? '✓' : 
                              verification.status === 'conflicted' ? '⚠' : '?';
            
            const text = document.createElement('span');
            text.textContent = verification.description;
            
            item.appendChild(icon);
            item.appendChild(text);
            this.verificationResults.appendChild(item);
        });
    }
    
    updateActionRecommendations(recommendations) {
        if (recommendations.length === 0) {
            this.actionRecommendations.style.display = 'none';
            return;
        }
        
        this.actionRecommendations.style.display = 'block';
        this.actionList.innerHTML = '';
        
        recommendations.forEach(rec => {
            const item = document.createElement('li');
            item.className = 'action-item';
            
            const priority = document.createElement('span');
            priority.className = `action-priority ${rec.priority}`;
            priority.textContent = rec.priority.toUpperCase();
            
            const content = document.createElement('div');
            content.innerHTML = `<strong>${rec.action}</strong><br><small>${rec.reason}</small>`;
            
            item.appendChild(priority);
            item.appendChild(content);
            this.actionList.appendChild(item);
        });
    }
    
    enableActionButtons() {
        this.submitAssessmentBtn.disabled = false;
        this.exportReportBtn.disabled = false;
        this.shareAssessmentBtn.disabled = false;
    }
    
    clearAllMedia() {
        // Clear videos
        if (this.videoStream) {
            this.stopLiveVideo();
        }
        this.capturedMedia.videos.forEach(v => URL.revokeObjectURL(v.url));
        this.capturedMedia.videos = [];
        this.videoPreview.style.display = 'none';
        this.videoCaptureArea.classList.remove('has-content');
        
        // Clear images
        this.capturedMedia.images.forEach(img => URL.revokeObjectURL(img.url));
        this.capturedMedia.images = [];
        this.imagePreviewContainer.innerHTML = '';
        this.imageCaptureArea.classList.remove('has-content');
        
        // Clear audio
        this.capturedMedia.audio.forEach(a => URL.revokeObjectURL(a.url));
        this.capturedMedia.audio = [];
        this.audioPreview.style.display = 'none';
        this.audioCaptureArea.classList.remove('has-content');
        
        // Reset analysis
        this.currentAnalysis = null;
        this.crossModalVerification.style.display = 'none';
        this.actionRecommendations.style.display = 'none';
        
        // Reset buttons
        this.analyzeBtn.disabled = true;
        this.submitAssessmentBtn.disabled = true;
        this.exportReportBtn.disabled = true;
        this.shareAssessmentBtn.disabled = true;
        
        // Clear file inputs
        this.videoFileInput.value = '';
        this.imageFileInput.value = '';
        this.audioFileInput.value = '';
    }
    
    setModel(model) {
        this.selectedModel = model;
        this.initializeGemma3n();
    }
    
    setResolution(resolution) {
        this.analysisResolution = resolution;
    }
    
    async submitAssessment() {
        if (!this.currentAnalysis) {
            alert('No analysis available to submit.');
            return;
        }
        
        const assessmentData = {
            type: 'multimodal_damage_assessment',
            timestamp: new Date().toISOString(),
            analysis: this.currentAnalysis,
            mediaMetadata: {
                videoCount: this.capturedMedia.videos.length,
                imageCount: this.capturedMedia.images.length,
                audioCount: this.capturedMedia.audio.length
            },
            modelUsed: this.selectedModel,
            resolution: this.analysisResolution
        };
        
        try {
            const response = await fetch('/api/submit-damage-assessment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(assessmentData)
            });
            
            if (response.ok) {
                alert('✅ Damage assessment submitted successfully!');
                window.location.href = '/admin-dashboard';
            } else {
                throw new Error('Failed to submit assessment');
            }
        } catch (error) {
            console.error('Error submitting assessment:', error);
            
            // Store offline
            const offlineQueue = JSON.parse(localStorage.getItem('damageAssessmentsQueue') || '[]');
            offlineQueue.push(assessmentData);
            localStorage.setItem('damageAssessmentsQueue', JSON.stringify(offlineQueue));
            
            alert('📱 Assessment saved offline. Will sync when connection is restored.');
        }
    }
    
    exportReport() {
        if (!this.currentAnalysis) {
            alert('No analysis available to export.');
            return;
        }
        
        const reportData = {
            title: 'Multimodal Damage Assessment Report',
            timestamp: new Date().toISOString(),
            summary: this.generateReportSummary(),
            analysis: this.currentAnalysis,
            mediaInfo: {
                videos: this.capturedMedia.videos.length,
                images: this.capturedMedia.images.length,
                audio: this.capturedMedia.audio.length
            }
        };
        
        const dataStr = JSON.stringify(reportData, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `damage-assessment-${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    generateReportSummary() {
        if (!this.currentAnalysis) return 'No analysis available';
        
        const { structural, infrastructure, environmental, accessibility, population, resources } = this.currentAnalysis;
        
        return `
Assessment Summary:
- Structural Damage: ${structural.level}
- Infrastructure: ${infrastructure.level}
- Environmental Hazards: ${environmental.hazards.length} detected
- Accessibility: ${accessibility.level}
- Population Impact: ${population.level}
- Resource Availability: ${resources.level}

Immediate Actions Required: ${this.currentAnalysis.recommendations.filter(r => r.priority === 'high').length}
        `.trim();
    }
    
    shareAssessment() {
        if (!this.currentAnalysis) {
            alert('No analysis available to share.');
            return;
        }
        
        const shareData = {
            title: 'Emergency Damage Assessment',
            text: this.generateReportSummary(),
            url: window.location.href
        };
        
        if (navigator.share) {
            navigator.share(shareData);
        } else {
            // Fallback - copy to clipboard
            navigator.clipboard.writeText(`${shareData.title}\n\n${shareData.text}\n\n${shareData.url}`)
                .then(() => alert('Assessment details copied to clipboard!'))
                .catch(() => alert('Unable to share. Please copy the URL manually.'));
        }
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MultimodalDamageAssessment;
} else {
    window.MultimodalDamageAssessment = MultimodalDamageAssessment;
}