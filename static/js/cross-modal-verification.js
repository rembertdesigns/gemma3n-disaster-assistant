class CrossModalVerificationEngine {
    constructor() {
        this.inputModalities = new Map(); // Stores the raw inputs for a specific verification run
        this.verificationResults = new Map(); // Stores results of each factor analysis for current run
        this.analysisHistory = []; // Stores past verification results
        this.confidenceThresholds = new Map(); // Configurable thresholds for each modality
        this.verificationFactors = new Map(); // Definition and weights of verification factors

        // Initialize core components
        this.initializeVerificationFactors();
        this.initializeConfidenceThresholds();
        this.initializeAnalysisModules();

        console.log('🔍 Cross-Modal Verification Engine initialized');
    }

    // ========================================================================
    // ⚙️ INITIALIZATION METHODS
    // ========================================================================

    initializeVerificationFactors() {
        // Defines core verification factors, their weights, and critical thresholds.
        // Scores for each factor will be normalized to 0-1.
        this.verificationFactors.set('temporal_consistency', {
            weight: 0.15,
            description: 'Timestamp and temporal sequence validation across inputs.',
            criticalThreshold: 0.7 // Below this score for the factor is critical
        });

        this.verificationFactors.set('location_verification', {
            weight: 0.20,
            description: 'Geographic and spatial consistency across inputs and external data.',
            criticalThreshold: 0.75
        });

        this.verificationFactors.set('content_authenticity', {
            weight: 0.25,
            description: 'Digital forensics, manipulation detection (e.g., Photoshop, deepfake).',
            criticalThreshold: 0.8
        });

        this.verificationFactors.set('source_credibility', {
            weight: 0.15,
            description: 'Source reputation, historical reliability, and internal trust scores.',
            criticalThreshold: 0.6
        });

        this.verificationFactors.set('technical_analysis', {
            weight: 0.15,
            description: 'Metadata (EXIF, file headers), format consistency, and integrity checks.',
            criticalThreshold: 0.7
        });

        this.verificationFactors.set('cross_modal_match', {
            weight: 0.10,
            description: 'Inter-modal consistency (e.g., audio matches video, text describes image).',
            criticalThreshold: 0.8
        });
        console.log('✅ Verification factors initialized.');
    }

    initializeConfidenceThresholds() {
        // Defines confidence tiers for each input modality.
        // These can be used to interpret the quality of raw analysis.
        this.confidenceThresholds.set('text', {
            excellent: 0.95, // AI model confidence
            good: 0.85,
            acceptable: 0.70,
            poor: 0.50
        });

        this.confidenceThresholds.set('image', {
            excellent: 0.98,
            good: 0.90,
            acceptable: 0.75,
            poor: 0.60
        });

        this.confidenceThresholds.set('audio', {
            excellent: 0.92,
            good: 0.82,
            acceptable: 0.68,
            poor: 0.45
        });

        this.confidenceThresholds.set('video', {
            excellent: 0.94,
            good: 0.85,
            acceptable: 0.72,
            poor: 0.55
        });
        console.log('✅ Confidence thresholds initialized.');
    }

    initializeAnalysisModules() {
        // Placeholder for integrating actual AI/analysis modules.
        // In a real system, these would be sophisticated APIs or local ML models.

        this.textAnalyzer = {
            analyzeSentiment: (text) => this._mockAnalyze('sentiment', text),
            extractEntities: (text) => this._mockAnalyze('entities', text),
            detectLanguage: (text) => this._mockAnalyze('language', text),
            validateTimestamps: (text) => this._mockAnalyze('text_timestamps', text),
            assessCredibility: (text) => this._mockAnalyze('text_credibility', text)
        };

        this.imageAnalyzer = {
            detectManipulation: (imageData) => this._mockAnalyze('img_manipulation', imageData),
            extractMetadata: (imageData) => this._mockAnalyze('img_metadata', imageData),
            analyzeContent: (imageData) => this._mockAnalyze('img_content', imageData),
            verifyLocation: (imageData) => this._mockAnalyze('img_location', imageData),
            detectDeepfakes: (imageData) => this._mockAnalyze('img_deepfake', imageData)
        };

        this.audioAnalyzer = {
            transcribeAudio: (audioData) => this._mockAnalyze('audio_transcription', audioData),
            analyzeVoiceStress: (audioData) => this._mockAnalyze('voice_stress', audioData),
            detectSynthetic: (audioData) => this._mockAnalyze('synthetic_audio', audioData),
            extractAmbientSound: (audioData) => this._mockAnalyze('ambient_sound', audioData),
            validateQuality: (audioData) => this._mockAnalyze('audio_quality', audioData)
        };

        this.videoAnalyzer = {
            extractFrames: (videoData) => this._mockAnalyze('video_frames', videoData),
            analyzeMotion: (videoData) => this._mockAnalyze('video_motion', videoData),
            detectEdits: (videoData) => this._mockAnalyze('video_edits', videoData),
            validateContinuity: (videoData) => this._mockAnalyze('video_continuity', videoData),
            extractAudio: (videoData) => this._mockAnalyze('video_audio_extract', videoData)
        };
        console.log('✅ Analysis modules initialized (mocked).');
    }

    // Simple mock analysis function for demonstration purposes
    _mockAnalyze(feature, data) {
        const confidence = Math.random() * (0.99 - 0.50) + 0.50; // Random confidence between 0.5 and 0.99
        return {
            result: `Mock analysis for ${feature} on ${typeof data === 'string' ? data.substring(0, 20) + '...' : 'binary data'}`,
            confidence: parseFloat(confidence.toFixed(2)),
            detected: Math.random() > 0.5 // Simulate detection
        };
    }

    // ========================================================================
    // 🚀 CORE VERIFICATION WORKFLOW
    // ========================================================================

    /**
     * Orchestrates the multi-modal verification process.
     * @param {Map<string, any>} inputs - A Map where keys are modality types (e.g., 'text', 'image', 'audio', 'video') and values are their corresponding data.
     * @returns {Promise<object>} The comprehensive verification report.
     */
    async processMultiModalInput(inputs) {
        if (!(inputs instanceof Map)) {
            throw new Error("Input must be a Map of modality types to data.");
        }
        if (inputs.size === 0) {
            throw new Error("No inputs provided for verification.");
        }

        const verification = {
            id: `verification-${Date.now()}`,
            timestamp: new Date().toISOString(),
            inputs: Array.from(inputs.entries()), // Convert Map to Array for storage/serialization
            modalAnalysis: new Map(), // Results of individual modality analysis
            crossModalAnalysis: new Map(), // Results of cross-modality checks
            factorScores: new Map(), // Scores for each defined verification factor
            verificationScore: 0, // Overall weighted score
            discrepancies: [], // List of identified inconsistencies or anomalies
            recommendations: [], // Actions based on the verification result
            verdict: 'pending' // Final verdict: 'verified', 'unverified', 'suspicious', 'inconclusive'
        };

        console.log('🔍 Starting multi-modal verification process...');
        const startTime = performance.now();

        // Step 1: Analyze each modality individually
        for (const [modalType, modalData] of inputs) {
            try {
                const modalAnalysis = await this.analyzeModality(modalType, modalData);
                verification.modalAnalysis.set(modalType, modalAnalysis);
                console.log(`✅ ${modalType} analysis complete: ${modalAnalysis.confidence}% confidence`);
            } catch (error) {
                console.error(`❌ Error analyzing ${modalType} modality:`, error);
                verification.discrepancies.push({
                    type: 'modality_analysis_failure',
                    modality: modalType,
                    details: error.message
                });
                verification.modalAnalysis.set(modalType, {
                    modalType,
                    quality: 'failed',
                    error: error.message,
                    confidence: 0
                });
            }
        }

        // Step 2: Perform cross-modal consistency analysis
        verification.crossModalAnalysis = await this.performCrossModalAnalysis(verification.modalAnalysis);

        // Step 3: Calculate verification factors' scores
        verification.factorScores = this.calculateVerificationFactors(verification);

        // Step 4: Compute overall verification score
        verification.verificationScore = this.calculateOverallScore(verification.factorScores);

        // Step 5: Identify discrepancies and anomalies based on factor scores and cross-modal analysis
        verification.discrepancies = this.identifyDiscrepancies(verification);

        // Step 6: Generate recommendations
        verification.recommendations = this.generateRecommendations(verification);

        // Step 7: Determine final verdict
        verification.verdict = this.determineVerdict(verification);

        const endTime = performance.now();
        verification.processingTimeMs = parseFloat((endTime - startTime).toFixed(2));

        // Store in history (e.g., for review in a dashboard)
        this.analysisHistory.push(verification);

        console.log(`🎯 Verification complete for ID ${verification.id}: ${verification.verificationScore.toFixed(2)}% - ${verification.verdict.toUpperCase()}. Took ${verification.processingTimeMs}ms.`);

        return verification;
    }

    /**
     * Analyzes a single modality (text, image, audio, video).
     * @param {string} modalType - The type of modality.
     * @param {any} modalData - The data for the modality.
     * @param {object} analysis - The analysis object to populate.
     * @returns {Promise<object>} The populated analysis object.
     */
    async analyzeModality(modalType, modalData, analysis) {
        try {
            switch (modalType) {
                case 'text':
                    const textSentiment = await this.textAnalyzer.analyzeSentiment(modalData);
                    const textEntities = await this.textAnalyzer.extractEntities(modalData);
                    const textLang = await this.textAnalyzer.detectLanguage(modalData);
                    analysis.features.set('sentiment', textSentiment.result);
                    analysis.features.set('entities', textEntities.result);
                    analysis.features.set('language', textLang.result);
                    analysis.confidence = textSentiment.confidence; // Example: take sentiment confidence
                    analysis.quality = textSentiment.confidence > this.confidenceThresholds.get('text').acceptable ? 'high' : 'low';
                    break;
                case 'image':
                    const imgManipulation = await this.imageAnalyzer.detectManipulation(modalData);
                    const imgMetadata = await this.imageAnalyzer.extractMetadata(modalData);
                    const imgContent = await this.imageAnalyzer.analyzeContent(modalData);
                    analysis.features.set('manipulation_detected', imgManipulation.detected);
                    analysis.features.set('metadata', imgMetadata.result);
                    analysis.features.set('content_labels', imgContent.result);
                    analysis.confidence = imgManipulation.confidence; // Example: take manipulation confidence
                    analysis.quality = imgManipulation.confidence > this.confidenceThresholds.get('image').acceptable ? 'high' : 'low';
                    if (imgManipulation.detected) analysis.anomalies.push('Image manipulation detected');
                    break;
                case 'audio':
                    const audioTranscript = await this.audioAnalyzer.transcribeAudio(modalData);
                    const voiceStress = await this.audioAnalyzer.analyzeVoiceStress(modalData);
                    analysis.features.set('transcript', audioTranscript.result);
                    analysis.features.set('voice_stress_level', voiceStress.result);
                    analysis.confidence = voiceStress.confidence; // Example: take voice stress confidence
                    analysis.quality = voiceStress.confidence > this.confidenceThresholds.get('audio').acceptable ? 'high' : 'low';
                    break;
                case 'video':
                    const videoEdits = await this.videoAnalyzer.detectEdits(modalData);
                    const videoMotion = await this.videoAnalyzer.analyzeMotion(modalData);
                    analysis.features.set('edits_detected', videoEdits.detected);
                    analysis.features.set('motion_patterns', videoMotion.result);
                    analysis.confidence = videoEdits.confidence; // Example: take video edits confidence
                    analysis.quality = videoEdits.confidence > this.confidenceThresholds.get('video').acceptable ? 'high' : 'low';
                    if (videoEdits.detected) analysis.anomalies.push('Video edits detected');
                    break;
                default:
                    throw new Error(`Unsupported modality type: ${modalType}`);
            }
            return analysis;
        } catch (error) {
            console.error(`Error during analysis of ${modalType}:`, error);
            analysis.quality = 'failed';
            analysis.error = error.message;
            analysis.confidence = 0;
            analysis.anomalies.push(`Failed to analyze: ${error.message}`);
            return analysis; // Return analysis with error info
        }
    }

    // ========================================================================
    // 🤝 CROSS-MODAL & FACTOR ANALYSIS
    // ========================================================================

    /**
     * Performs consistency checks across different modalities.
     * @param {Map<string, object>} modalAnalyses - Map of results from individual modality analysis.
     * @returns {Promise<Map<string, object>>} Results of cross-modal checks.
     */
    async performCrossModalAnalysis(modalAnalyses) {
        const crossChecks = new Map();

        // Example: Text-Image Consistency
        if (modalAnalyses.has('text') && modalAnalyses.has('image')) {
            const textFeatures = modalAnalyses.get('text').features;
            const imageFeatures = modalAnalyses.get('image').features;
            // Simulate AI matching text entities with image content labels
            const matchConfidence = this._mockAnalyze('text_image_match', '').confidence;
            crossChecks.set('text_image_consistency', {
                confidence: matchConfidence,
                details: `Text entities vs. image content match confidence: ${matchConfidence}`,
                consistent: matchConfidence > 0.7
            });
            if (matchConfidence <= 0.7) crossChecks.get('text_image_consistency').anomaly = true;
        }

        // Example: Temporal Consistency (timestamps from text, image metadata, etc.)
        const timestamps = [];
        if (modalAnalyses.has('text') && modalAnalyses.get('text').features.has('text_timestamps')) {
            timestamps.push(modalAnalyses.get('text').features.get('text_timestamps').result);
        }
        if (modalAnalyses.has('image') && modalAnalyses.get('image').features.has('metadata')) {
            // Extract EXIF date/time
            timestamps.push(this._mockAnalyze('extract_exif_date', '').result);
        }
        // If multiple timestamps, compare them for consistency
        if (timestamps.length > 1) {
            const tempConsistency = this._mockAnalyze('temporal_diff', timestamps).confidence;
            crossChecks.set('temporal_consistency_score', {
                confidence: tempConsistency,
                details: `Temporal consistency across inputs: ${tempConsistency}`,
                consistent: tempConsistency > 0.8
            });
            if (tempConsistency <= 0.8) crossChecks.get('temporal_consistency_score').anomaly = true;
        }


        // Add more cross-modal checks (e.g., Audio-Video sync, Location consistency across text/image/GPS)
        // ... (complex logic would go here) ...

        console.log('✅ Cross-modal analysis complete.');
        return crossChecks;
    }

    /**
     * Calculates scores for each defined verification factor.
     * @param {object} verification - The main verification object.
     * @returns {Map<string, number>} Scores for each factor (0-1).
     */
    calculateVerificationFactors(verification) {
        const factorScores = new Map();
        const modalAnalyses = verification.modalAnalysis;
        const crossModalAnalyses = verification.crossModalAnalysis;

        // Temporal Consistency
        let temporalScore = 0.5; // Default score
        if (crossModalAnalyses.has('temporal_consistency_score')) {
            temporalScore = crossModalAnalyses.get('temporal_consistency_score').confidence;
        } else if (modalAnalyses.has('text') && modalAnalyses.get('text').features.has('text_timestamps')) {
            temporalScore = modalAnalyses.get('text').features.get('text_timestamps').confidence;
        }
        factorScores.set('temporal_consistency', temporalScore);

        // Location Verification
        let locationScore = 0.5;
        if (modalAnalyses.has('image') && modalAnalyses.get('image').features.has('img_location')) {
            locationScore = modalAnalyses.get('image').features.get('img_location').confidence;
        }
        // Also consider text-based locations and external geo-fencing data
        factorScores.set('location_verification', locationScore);

        // Content Authenticity (e.g., Image manipulation, synthetic audio/video)
        let authenticityScore = 1.0; // Start high, deduct for anomalies
        if (modalAnalyses.has('image') && modalAnalyses.get('image').features.get('manipulation_detected')) {
            authenticityScore -= 0.5; // Significant deduction
        }
        if (modalAnalyses.has('video') && modalAnalyses.get('video').features.get('edits_detected')) {
            authenticityScore -= 0.4;
        }
        if (modalAnalyses.has('audio') && modalAnalyses.get('audio').features.get('synthetic_audio')) {
            authenticityScore -= 0.6;
        }
        authenticityScore = Math.max(0, authenticityScore); // Ensure non-negative
        factorScores.set('content_authenticity', authenticityScore);

        // Source Credibility (placeholder - would rely on user profiles, history, external data)
        factorScores.set('source_credibility', verification.sourceCredibilityScore || 0.7); // Assume 0.7 if not provided

        // Technical Analysis (metadata consistency, file integrity)
        let technicalScore = 1.0;
        if (modalAnalyses.has('image') && modalAnalyses.get('image').features.get('metadata') &&
            modalAnalyses.get('image').features.get('metadata').result === 'inconsistent') { // Example check
            technicalScore -= 0.3;
        }
        factorScores.set('technical_analysis', technicalScore);

        // Cross-Modal Match
        let crossModalScore = 1.0;
        if (crossModalAnalyses.has('text_image_consistency') && !crossModalAnalyses.get('text_image_consistency').consistent) {
            crossModalScore -= 0.5;
        }
        // Deduct based on any detected cross-modal anomalies
        crossModalScore = Math.max(0, crossModalScore);
        factorScores.set('cross_modal_match', crossModalScore);

        console.log('✅ Verification factors calculated.');
        return factorScores;
    }

    /**
     * Calculates the overall weighted verification score.
     * @param {Map<string, number>} factorScores - Scores for each verification factor.
     * @returns {number} The overall score (0-100).
     */
    calculateOverallScore(factorScores) {
        let totalWeightedScore = 0;
        let totalWeight = 0;

        for (const [factorName, factorConfig] of this.verificationFactors.entries()) {
            const score = factorScores.get(factorName);
            if (score !== undefined) {
                totalWeightedScore += score * factorConfig.weight;
                totalWeight += factorConfig.weight;
            }
        }

        // Normalize to 0-100 scale
        const overallScore = totalWeight > 0 ? (totalWeightedScore / totalWeight) * 100 : 0;
        console.log(`✅ Overall verification score: ${overallScore.toFixed(2)}%`);
        return parseFloat(overallScore.toFixed(2));
    }

    /**
     * Identifies discrepancies based on analysis results and factor scores.
     * @param {object} verification - The main verification object.
     * @returns {Array<string>} A list of identified discrepancies.
     */
    identifyDiscrepancies(verification) {
        const discrepancies = [];

        // Add anomalies from individual modality analysis
        for (const [modalType, analysis] of verification.modalAnalysis.entries()) {
            if (analysis.anomalies && analysis.anomalies.length > 0) {
                analysis.anomalies.forEach(anomaly => discrepancies.push(`${modalType.capitalize()}: ${anomaly}`));
            }
            if (analysis.quality === 'failed') {
                discrepancies.push(`${modalType.capitalize()} analysis failed: ${analysis.error || 'unknown error'}`);
            }
        }

        // Add anomalies from cross-modal analysis
        for (const [checkName, checkResult] of verification.crossModalAnalysis.entries()) {
            if (checkResult.anomaly) {
                discrepancies.push(`Cross-Modal Discrepancy (${checkName}): ${checkResult.details}`);
            }
        }

        // Add discrepancies based on low factor scores
        for (const [factorName, score] of verification.factorScores.entries()) {
            const factorConfig = this.verificationFactors.get(factorName);
            if (factorConfig && score < factorConfig.criticalThreshold) {
                discrepancies.push(`${factorConfig.description} score is low (${score.toFixed(2)} < ${factorConfig.criticalThreshold}). Potential issue.`);
            }
        }

        return discrepancies;
    }

    /**
     * Generates recommendations based on the verification result.
     * @param {object} verification - The main verification object.
     * @returns {Array<string>} A list of recommendations.
     */
    generateRecommendations(verification) {
        const recommendations = [];

        if (verification.verdict === 'verified') {
            recommendations.push('Proceed with emergency response based on high confidence.');
            if (verification.discrepancies.length > 0) {
                recommendations.push('Minor discrepancies noted, but not critical. Monitor for updates.');
            }
        } else if (verification.verdict === 'suspicious') {
            recommendations.push('Review report manually due to suspicious elements.');
            verification.discrepancies.forEach(d => recommendations.push(`Investigate: ${d}`));
            recommendations.push('Request additional information or alternative inputs if possible.');
        } else if (verification.verdict === 'unverified') {
            recommendations.push('Exercise extreme caution. Do not base critical decisions solely on this report.');
            recommendations.push('Prioritize independent verification via other means.');
            recommendations.push('Flag report for manual review and potential rejection.');
        } else if (verification.verdict === 'inconclusive') {
            recommendations.push('Insufficient data for a definitive verdict. Seek more information.');
            recommendations.push('Treat as potentially unreliable until further evidence is gathered.');
        }

        if (verification.processingTimeMs > 5000) { // Example: If analysis took too long
            recommendations.push('Analysis was slow. Check system load or input size.');
        }

        return recommendations;
    }

    /**
     * Determines the final verdict based on the overall verification score and discrepancies.
     * @param {object} verification - The main verification object.
     * @returns {string} The final verdict ('verified', 'unverified', 'suspicious', 'inconclusive').
     */
    determineVerdict(verification) {
        const score = verification.verificationScore;
        const discrepanciesCount = verification.discrepancies.length;
        const totalFactors = this.verificationFactors.size;
        const failedFactors = Array.from(verification.factorScores.entries())
                                .filter(([name, score]) => score < this.verificationFactors.get(name).criticalThreshold)
                                .length;

        if (score >= 90 && discrepanciesCount === 0) {
            return 'verified';
        } else if (score >= 70 && failedFactors < 1) {
            // High score, but maybe minor discrepancies
            return 'verified_with_minor_issues';
        } else if (score >= 50 && failedFactors < totalFactors / 2) {
            // Moderate score, some discrepancies
            return 'suspicious';
        } else if (score > 0 && failedFactors >= totalFactors / 2) {
            // Low score, many critical issues or analytical failures
            return 'unverified';
        } else {
            // If score is very low, or many analyses failed
            const failedModalities = Array.from(verification.modalAnalysis.values())
                                        .filter(ma => ma.quality === 'failed').length;
            if (failedModalities > verification.modalAnalysis.size / 2) {
                return 'inconclusive'; // Cannot verify due to lack of data/analysis
            }
            return 'unverified'; // Default to unverified if nothing else fits
        }
    }
}

// Helper to capitalize first letter of a string (for display in discrepancies)
String.prototype.capitalize = function() {
    return this.charAt(0).toUpperCase() + this.slice(1);
}

// ============================================================================
// USAGE EXAMPLE (for testing/demonstration)
// ============================================================================

async function runVerificationDemo() {
    const engine = new CrossModalVerificationEngine();

    console.log('\n--- Running Demo 1: Consistent Report ---');
    const inputs1 = new Map();
    inputs1.set('text', 'Fire reported at downtown warehouse, heavy smoke. Timestamp: 2025-07-09T10:00:00Z. Location: 34.0522, -118.2437.');
    inputs1.set('image', 'mock_fire_image_data_jpg');
    inputs1.set('audio', 'mock_audio_siren_and_voice_data');
    inputs1.set('video', 'mock_video_fire_clip_mp4');
    
    // Simulate high confidence results for demo 1
    engine.textAnalyzer.analyzeSentiment = (text) => ({ result: 'urgent', confidence: 0.98 });
    engine.imageAnalyzer.detectManipulation = (img) => ({ detected: false, confidence: 0.95 });
    engine.audioAnalyzer.analyzeVoiceStress = (aud) => ({ result: 'high_stress', confidence: 0.92 });
    engine.videoAnalyzer.detectEdits = (vid) => ({ detected: false, confidence: 0.94 });
    engine._mockAnalyze = (feature) => ({ result: 'mock_result', confidence: Math.random() * (0.99 - 0.85) + 0.85, detected: false }); // Higher random confidence

    const report1 = await engine.processMultiModalInput(inputs1);
    console.log('Final Report 1:', JSON.stringify(report1, null, 2));


    console.log('\n--- Running Demo 2: Suspicious Report ---');
    const inputs2 = new Map();
    inputs2.set('text', 'Minor incident, small fire, no danger. Location: 34.0530, -118.2445.');
    inputs2.set('image', 'mock_manipulated_image_data_png'); // Simulate manipulated image
    inputs2.set('audio', 'mock_low_quality_audio_data'); // Simulate low quality audio
    
    // Simulate mixed/lower confidence results for demo 2
    engine.textAnalyzer.analyzeSentiment = (text) => ({ result: 'neutral', confidence: 0.65 });
    engine.imageAnalyzer.detectManipulation = (img) => ({ detected: true, confidence: 0.40 }); // Low confidence, detected manipulation
    engine.audioAnalyzer.analyzeVoiceStress = (aud) => ({ result: 'unknown', confidence: 0.30 }); // Very low confidence audio
    engine._mockAnalyze = (feature) => ({ result: 'mock_result', confidence: Math.random() * (0.7 - 0.3) + 0.3, detected: Math.random() > 0.7 }); // Lower random confidence, more detections

    const report2 = await engine.processMultiModalInput(inputs2);
    console.log('Final Report 2:', JSON.stringify(report2, null, 2));

    console.log('\n--- Running Demo 3: Inconclusive Report (Missing Modalities) ---');
    const inputs3 = new Map();
    inputs3.set('text', 'Just some text, no other media.');
    
    // Reset mock analysis for demo 3
    engine._mockAnalyze = (feature) => ({ result: 'mock_result', confidence: Math.random() * (0.9 - 0.8) + 0.8, detected: false });

    const report3 = await engine.processMultiModalInput(inputs3);
    console.log('Final Report 3:', JSON.stringify(report3, null, 2));
}

// Call the demo function to see it in action
// runVerificationDemo(); // Uncomment this line to run the demo when the script loads.