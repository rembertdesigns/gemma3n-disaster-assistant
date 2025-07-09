/**
 * Context Intelligence Dashboard - Gemma 3n 128K Context Processing
 * Handles deep situation analysis and comprehensive data synthesis
 */

class ContextIntelligenceDashboard {
    constructor() {
        this.selectedDataSources = new Set(['emergency-reports']);
        this.contextUsage = 0;
        this.maxContextTokens = 128000;
        this.currentAnalysis = null;
        this.selectedModel = 'gemma-3n-4b';
        this.activeTab = 'insights';
        this.timelinePeriod = '24h';
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadInitialData();
        this.initializeGemma3n();
    }
    
    initializeElements() {
        // Context elements
        this.contextFill = document.getElementById('contextFill');
        this.contextUsage = document.getElementById('contextUsage');
        this.analyzeContextBtn = document.getElementById('analyzeContextBtn');
        
        // Analysis elements
        this.insightGrid = document.getElementById('insightGrid');
        this.timelineEvents = document.getElementById('timelineEvents');
        this.correlationMatrix = document.getElementById('correlationMatrix');
        this.synthesisContent = document.getElementById('synthesisContent');
        this.keyFindings = document.getElementById('keyFindings');
        this.findingsList = document.getElementById('findingsList');
        
        // Action buttons
        this.deepAnalysisBtn = document.getElementById('deepAnalysisBtn');
        this.exportInsightsBtn = document.getElementById('exportInsightsBtn');
        this.shareInsightsBtn = document.getElementById('shareInsightsBtn');
        this.clearContextBtn = document.getElementById('clearContextBtn');
        
        // Processing elements
        this.processingIntelligence = document.getElementById('processingIntelligence');
        this.processingStatus = document.getElementById('processingStatus');
        this.tokensProcessed = document.getElementById('tokensProcessed');
        this.patternsFound = document.getElementById('patternsFound');
        this.correlationsDetected = document.getElementById('correlationsDetected');
        
        // Data source counters
        this.dataSourceCounters = {
            'emergency-reports': document.getElementById('emergencyCount'),
            'weather-data': document.getElementById('weatherCount'),
            'infrastructure': document.getElementById('infrastructureCount'),
            'population': document.getElementById('populationCount'),
            'resources': document.getElementById('resourceCount'),
            'social-media': document.getElementById('socialCount'),
            'historical': document.getElementById('historicalCount')
        };
        
        this.dataSourceSizes = {
            'emergency-reports': document.getElementById('emergencySize'),
            'weather-data': document.getElementById('weatherSize'),
            'infrastructure': document.getElementById('infrastructureSize'),
            'population': document.getElementById('populationSize'),
            'resources': document.getElementById('resourceSize'),
            'social-media': document.getElementById('socialSize'),
            'historical': document.getElementById('historicalSize')
        };
    }
    
    async initializeGemma3n() {
        try {
            if (window.EdgeAI) {
                await window.EdgeAI.loadContextModel(this.selectedModel);
                console.log('✅ Gemma 3n context model loaded');
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
                aiStatusText.textContent = '🧠 Gemma 3n 128K Context Ready';
                break;
            case 'processing':
                aiStatusDot.className = 'ai-status-dot loading';
                aiStatusText.textContent = '🧠 Processing 128K Context...';
                break;
            case 'fallback':
                aiStatusDot.className = 'ai-status-dot loading';
                aiStatusText.textContent = '🧠 Using Fallback Context Processing';
                break;
            case 'error':
                aiStatusDot.className = 'ai-status-dot error';
                aiStatusText.textContent = '🧠 Context AI Unavailable';
                break;
        }
    }
    
    setupEventListeners() {
        // Main analysis button
        this.analyzeContextBtn.addEventListener('click', () => {
            this.performContextAnalysis();
        });
        
        // Action buttons
        this.deepAnalysisBtn.addEventListener('click', () => {
            this.performDeepAnalysis();
        });
        
        this.exportInsightsBtn.addEventListener('click', () => {
            this.exportInsights();
        });
        
        this.shareInsightsBtn.addEventListener('click', () => {
            this.shareInsights();
        });
        
        this.clearContextBtn.addEventListener('click', () => {
            this.clearContext();
        });
        
        // Data source selection
        document.querySelectorAll('.data-source-item').forEach(item => {
            item.addEventListener('click', (e) => {
                this.selectDataSource(e.currentTarget.dataset.source);
            });
        });
    }
    
    async loadInitialData() {
        try {
            // Load data source counts and sizes
            const dataSources = [
                { id: 'emergency-reports', endpoint: '/api/recent-reports?limit=50' },
                { id: 'weather-data', endpoint: '/api/weather-data', fallback: { count: 24, size: '15 KB' }},
                { id: 'infrastructure', endpoint: '/api/infrastructure-status', fallback: { count: 156, size: '8 KB' }},
                { id: 'population', endpoint: '/api/population-data', fallback: { count: 12, size: '22 KB' }},
                { id: 'resources', endpoint: '/api/resource-inventory', fallback: { count: 89, size: '12 KB' }},
                { id: 'social-media', endpoint: '/api/social-feeds', fallback: { count: 342, size: '45 KB' }},
                { id: 'historical', endpoint: '/api/historical-patterns', fallback: { count: 1247, size: '156 KB' }}
            ];
            
            for (const source of dataSources) {
                try {
                    let data;
                    if (source.id === 'emergency-reports') {
                        const response = await fetch(source.endpoint);
                        if (response.ok) {
                            data = await response.json();
                            this.updateDataSourceDisplay(source.id, data.count || 0, '8 KB');
                        } else {
                            throw new Error('Failed to fetch');
                        }
                    } else {
                        // Use fallback data for other sources
                        data = source.fallback;
                        this.updateDataSourceDisplay(source.id, data.count, data.size);
                    }
                } catch (error) {
                    console.warn(`⚠️ Failed to load ${source.id}, using fallback`);
                    const fallback = source.fallback || { count: 0, size: '0 KB' };
                    this.updateDataSourceDisplay(source.id, fallback.count, fallback.size);
                }
            }
            
            this.updateContextUsage();
            
        } catch (error) {
            console.error('❌ Error loading initial data:', error);
        }
    }
    
    updateDataSourceDisplay(sourceId, count, size) {
        if (this.dataSourceCounters[sourceId]) {
            this.dataSourceCounters[sourceId].textContent = count;
        }
        if (this.dataSourceSizes[sourceId]) {
            this.dataSourceSizes[sourceId].textContent = size;
        }
    }
    
    selectDataSource(sourceId) {
        if (this.selectedDataSources.has(sourceId)) {
            this.selectedDataSources.delete(sourceId);
        } else {
            this.selectedDataSources.add(sourceId);
        }
        
        // Update UI
        document.querySelectorAll('.data-source-item').forEach(item => {
            const isSelected = this.selectedDataSources.has(item.dataset.source);
            item.classList.toggle('active', isSelected);
        });
        
        this.updateContextUsage();
    }
    
    updateContextUsage() {
        // Calculate estimated token usage based on selected data sources
        const tokenEstimates = {
            'emergency-reports': 15000,
            'weather-data': 8000,
            'infrastructure': 12000,
            'population': 6000,
            'resources': 4000,
            'social-media': 25000,
            'historical': 45000
        };
        
        let totalTokens = 0;
        this.selectedDataSources.forEach(sourceId => {
            totalTokens += tokenEstimates[sourceId] || 0;
        });
        
        const usagePercent = Math.min((totalTokens / this.maxContextTokens) * 100, 100);
        
        this.contextFill.style.width = `${usagePercent}%`;
        this.contextUsage.textContent = `${Math.round(usagePercent)}%`;
        
        // Update context fill color based on usage
        if (usagePercent > 90) {
            this.contextFill.style.background = '#ef4444';
        } else if (usagePercent > 70) {
            this.contextFill.style.background = '#f59e0b';
        } else {
            this.contextFill.style.background = '#10b981';
        }
        
        this.currentTokenCount = totalTokens;
    }
    
    async performContextAnalysis() {
        if (this.selectedDataSources.size === 0) {
            alert('Please select at least one data source for analysis.');
            return;
        }
        
        this.showProcessingIntelligence();
        this.updateAIStatus('processing');
        
        try {
            // Prepare analysis data
            const analysisData = {
                dataSources: Array.from(this.selectedDataSources),
                contextTokens: this.currentTokenCount,
                model: this.selectedModel,
                analysisType: 'comprehensive',
                timestamp: new Date().toISOString()
            };
            
            // Perform Gemma 3n context analysis
            const analysis = await this.analyzeWithGemma3n(analysisData);
            
            // Display results
            this.displayContextResults(analysis);
            this.enableActionButtons();
            
        } catch (error) {
            console.error('Error during context analysis:', error);
            alert('Analysis failed. Please try again.');
        } finally {
            this.hideProcessingIntelligence();
            this.updateAIStatus('ready');
        }
    }
    
    showProcessingIntelligence() {
        this.processingIntelligence.style.display = 'block';
        
        // Simulate processing steps
        const steps = [
            'Loading data sources...',
            'Processing emergency reports...',
            'Analyzing weather patterns...',
            'Cross-referencing infrastructure data...',
            'Identifying correlations...',
            'Generating insights...',
            'Synthesizing findings...'
        ];
        
        let stepIndex = 0;
        let tokenCount = 0;
        let patternCount = 0;
        let correlationCount = 0;
        
        const processingInterval = setInterval(() => {
            if (stepIndex < steps.length) {
                this.processingStatus.textContent = steps[stepIndex];
                stepIndex++;
                
                // Update counters
                tokenCount += Math.floor(Math.random() * 5000) + 1000;
                this.tokensProcessed.textContent = tokenCount.toLocaleString();
                
                if (stepIndex > 3) {
                    patternCount += Math.floor(Math.random() * 3) + 1;
                    this.patternsFound.textContent = patternCount;
                }
                
                if (stepIndex > 4) {
                    correlationCount += Math.floor(Math.random() * 2) + 1;
                    this.correlationsDetected.textContent = correlationCount;
                }
            } else {
                clearInterval(processingInterval);
                this.processingStatus.textContent = 'Analysis complete';
            }
        }, 1000);
    }
    
    hideProcessingIntelligence() {
        this.processingIntelligence.style.display = 'none';
    }
    
    async analyzeWithGemma3n(analysisData) {
        // Simulate Gemma 3n context analysis with 128K token processing
        await new Promise(resolve => setTimeout(resolve, 8000));
        
        // Generate comprehensive analysis
        return this.simulateContextAnalysis(analysisData);
    }
    
    simulateContextAnalysis(analysisData) {
        const insights = [];
        const correlations = [];
        const timelineEvents = [];
        
        // Generate insights based on selected data sources
        if (analysisData.dataSources.includes('emergency-reports')) {
            insights.push({
                type: 'trend',
                title: 'Emergency Report Volume Spike',
                content: 'Emergency reports have increased by 34% in the past 48 hours, particularly concentrated in downtown areas.',
                confidence: 0.92,
                priority: 'high',
                metrics: { change: '+34%', timeframe: '48h', locations: 'Downtown' }
            });
        }
        
        if (analysisData.dataSources.includes('weather-data')) {
            insights.push({
                type: 'correlation',
                title: 'Weather-Emergency Correlation',
                content: 'Strong correlation (r=0.78) between precipitation levels and infrastructure-related emergency reports.',
                confidence: 0.87,
                priority: 'medium',
                metrics: { correlation: '0.78', factor: 'Precipitation', impact: 'Infrastructure' }
            });
        }
        
        if (analysisData.dataSources.includes('historical')) {
            insights.push({
                type: 'prediction',
                title: 'Historical Pattern Recognition',
                content: 'Current conditions match historical patterns preceding major flooding events (87% similarity).',
                confidence: 0.89,
                priority: 'critical',
                metrics: { similarity: '87%', event_type: 'Flooding', timeframe: 'Next 72h' }
            });
        }
        
        if (analysisData.dataSources.includes('social-media')) {
            insights.push({
                type: 'anomaly',
                title: 'Social Media Sentiment Shift',
                content: 'Unusual spike in anxiety-related social media posts, 245% above baseline in affected areas.',
                confidence: 0.83,
                priority: 'medium',
                metrics: { change: '+245%', sentiment: 'Anxiety', baseline: 'Normal' }
            });
        }
        
        // Generate correlations
        correlations.push(
            { source1: 'Weather', source2: 'Emergency Reports', strength: 0.78, type: 'positive' },
            { source1: 'Infrastructure', source2: 'Population Density', strength: 0.65, type: 'positive' },
            { source1: 'Social Media', source2: 'Emergency Reports', strength: 0.71, type: 'positive' },
            { source1: 'Historical Patterns', source2: 'Current Events', strength: 0.87, type: 'positive' }
        );
        
        // Generate timeline events
        const now = new Date();
        for (let i = 0; i < 10; i++) {
            const eventTime = new Date(now.getTime() - (i * 2 * 60 * 60 * 1000)); // Every 2 hours
            timelineEvents.push({
                time: eventTime,
                severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)],
                description: `Event ${i + 1}: ${['Infrastructure alert', 'Weather warning', 'Emergency report', 'Resource update'][Math.floor(Math.random() * 4)]}`,
                position: (i / 9) * 100 // Position percentage
            });
        }
        
        // Generate synthesis
        const synthesis = {
            summary: `Comprehensive analysis of ${analysisData.dataSources.length} data sources reveals heightened emergency activity with strong weather correlations. Historical pattern matching suggests elevated risk for flooding events. Recommend increased monitoring and resource pre-positioning.`,
            keyFindings: [
                { priority: 'critical', text: 'Historical pattern match for flooding (87% similarity) - recommend immediate preparation' },
                { priority: 'high', text: 'Emergency reports up 34% in downtown areas - deploy additional resources' },
                { priority: 'medium', text: 'Strong weather-infrastructure correlation detected - monitor vulnerable areas' },
                { priority: 'low', text: 'Social media sentiment monitoring effective for early warning detection' }
            ],
            recommendations: [
                'Increase emergency response team readiness',
                'Pre-position flood response equipment',
                'Issue public preparedness advisories',
                'Monitor infrastructure in high-risk areas'
            ]
        };
        
        return {
            insights,
            correlations,
            timelineEvents,
            synthesis,
            metadata: {
                tokensProcessed: analysisData.contextTokens,
                dataSources: analysisData.dataSources,
                analysisTime: new Date().toISOString(),
                model: analysisData.model
            }
        };
    }
    
    displayContextResults(analysis) {
        this.currentAnalysis = analysis;
        
        // Display insights
        this.displayInsights(analysis.insights);
        
        // Display timeline
        this.displayTimeline(analysis.timelineEvents);
        
        // Display correlations
        this.displayCorrelations(analysis.correlations);
        
        // Display synthesis
        this.displaySynthesis(analysis.synthesis);
    }
    
    displayInsights(insights) {
        this.insightGrid.innerHTML = '';
        
        insights.forEach(insight => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            
            card.innerHTML = `
                <div class="insight-header">
                    <div class="insight-icon ${insight.type}">
                        ${this.getInsightIcon(insight.type)}
                    </div>
                    <div class="insight-title">${insight.title}</div>
                    <div class="insight-confidence">${Math.round(insight.confidence * 100)}%</div>
                </div>
                <div class="insight-content">${insight.content}</div>
                <div class="insight-metrics">
                    ${Object.entries(insight.metrics).map(([key, value]) => `
                        <div class="metric">
                            <div class="metric-value">${value}</div>
                            <div class="metric-label">${key.replace(/_/g, ' ')}</div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            this.insightGrid.appendChild(card);
        });
    }
    
    getInsightIcon(type) {
        const icons = {
            trend: '📈',
            correlation: '🔗',
            prediction: '🔮',
            anomaly: '⚠️'
        };
        return icons[type] || '💡';
    }
    
    displayTimeline(events) {
        this.timelineEvents.innerHTML = '';
        
        events.forEach(event => {
            const eventElement = document.createElement('div');
            eventElement.className = `timeline-event ${event.severity}`;
            eventElement.style.left = `${event.position}%`;
            eventElement.title = `${event.description} - ${event.time.toLocaleString()}`;
            
            this.timelineEvents.appendChild(eventElement);
        });
    }
    
    displayCorrelations(correlations) {
        this.correlationMatrix.innerHTML = '';
        
        const matrixGrid = document.createElement('div');
        matrixGrid.className = 'matrix-grid';
        
        correlations.forEach(correlation => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            
            const strength = Math.round(correlation.strength * 100);
            cell.textContent = `${strength}%`;
            cell.title = `${correlation.source1} ↔ ${correlation.source2}: ${strength}% correlation`;
            
            // Color based on strength
            if (strength > 80) {
                cell.style.background = '#dc2626';
            } else if (strength > 60) {
                cell.style.background = '#f59e0b';
            } else if (strength > 40) {
                cell.style.background = '#10b981';
            } else {
                cell.style.background = '#6b7280';
            }
            
            matrixGrid.appendChild(cell);
        });
        
        this.correlationMatrix.appendChild(matrixGrid);
    }
    
    displaySynthesis(synthesis) {
        this.synthesisContent.textContent = synthesis.summary;
        
        // Display key findings
        this.keyFindings.style.display = 'block';
        this.findingsList.innerHTML = '';
        
        synthesis.keyFindings.forEach(finding => {
            const listItem = document.createElement('li');
            listItem.className = 'finding-item';
            
            listItem.innerHTML = `
                <div class="finding-priority ${finding.priority}">${finding.priority.toUpperCase()}</div>
                <div>${finding.text}</div>
            `;
            
            this.findingsList.appendChild(listItem);
        });
    }
    
    enableActionButtons() {
        this.deepAnalysisBtn.disabled = false;
        this.exportInsightsBtn.disabled = false;
        this.shareInsightsBtn.disabled = false;
    }
    
    updateTimelinePeriod(period) {
        this.timelinePeriod = period;
        // In a real implementation, this would reload timeline data
        console.log(`Timeline period changed to: ${period}`);
    }
    
    async performDeepAnalysis() {
        if (!this.currentAnalysis) {
            alert('Please run context analysis first.');
            return;
        }
        
        try {
            // Submit analysis to server
            const response = await fetch('/api/context-analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    analysis_type: 'deep',
                    data_sources: Array.from(this.selectedDataSources),
                    insights: this.currentAnalysis.insights,
                    synthesis: this.currentAnalysis.synthesis
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                alert('✅ Deep analysis completed and saved!');
                console.log('Deep analysis result:', result);
            } else {
                throw new Error('Failed to submit analysis');
            }
        } catch (error) {
            console.error('Error performing deep analysis:', error);
            alert('❌ Deep analysis failed. Please try again.');
        }
    }
    
    exportInsights() {
        if (!this.currentAnalysis) {
            alert('No analysis available to export.');
            return;
        }
        
        const exportData = {
            title: 'Context Intelligence Analysis Report',
            timestamp: new Date().toISOString(),
            metadata: this.currentAnalysis.metadata,
            insights: this.currentAnalysis.insights,
            correlations: this.currentAnalysis.correlations,
            synthesis: this.currentAnalysis.synthesis,
            dataSources: Array.from(this.selectedDataSources)
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `context-analysis-${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    shareInsights() {
        if (!this.currentAnalysis) {
            alert('No analysis available to share.');
            return;
        }
        
        const shareText = `Context Intelligence Analysis Report

Key Insights: ${this.currentAnalysis.insights.length} patterns identified
Data Sources: ${Array.from(this.selectedDataSources).join(', ')}
Tokens Processed: ${this.currentAnalysis.metadata.tokensProcessed.toLocaleString()}

Summary: ${this.currentAnalysis.synthesis.summary}

Generated by Gemma 3n Context Intelligence Dashboard`;
        
        if (navigator.share) {
            navigator.share({
                title: 'Context Intelligence Analysis',
                text: shareText,
                url: window.location.href
            });
        } else {
            navigator.clipboard.writeText(shareText)
                .then(() => alert('Analysis summary copied to clipboard!'))
                .catch(() => alert('Unable to share. Please copy manually.'));
        }
    }
    
    clearContext() {
        if (confirm('Are you sure you want to clear the current context analysis?')) {
            this.currentAnalysis = null;
            this.selectedDataSources.clear();
            this.selectedDataSources.add('emergency-reports'); // Keep default selection
            
            // Update UI
            document.querySelectorAll('.data-source-item').forEach(item => {
                item.classList.remove('active');
            });
            document.querySelector('[data-source="emergency-reports"]').classList.add('active');
            
            // Clear results
            this.insightGrid.innerHTML = '<p>Run context analysis to generate insights...</p>';
            this.timelineEvents.innerHTML = '';
            this.correlationMatrix.innerHTML = '<p>No correlations analyzed yet.</p>';
            this.synthesisContent.textContent = 'Click "Analyze Context" to generate comprehensive situation analysis using all available data sources...';
            this.keyFindings.style.display = 'none';
            
            // Reset buttons
            this.deepAnalysisBtn.disabled = true;
            this.exportInsightsBtn.disabled = true;
            this.shareInsightsBtn.disabled = true;
            
            this.updateContextUsage();
        }
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ContextIntelligenceDashboard;
} else {
    window.ContextIntelligenceDashboard = ContextIntelligenceDashboard;
}