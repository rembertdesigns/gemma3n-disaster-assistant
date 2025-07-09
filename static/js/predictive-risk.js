// Predictive Risk Modeling Engine
// Advanced emergency forecasting with AI-enhanced analysis

class PredictiveRiskEngine {
    constructor() {
      this.riskModels = new Map();
      this.realTimeData = new Map();
      this.contextFactors = new Map();
      this.forecastCache = new Map();
      this.updateInterval = null;
      
      // Initialize risk models
      this.initializeRiskModels();
      this.initializeContextFactors();
      
      console.log('🔮 Predictive Risk Engine initialized');
    }
    
    initializeRiskModels() {
      // Seismic risk model
      this.riskModels.set('earthquake', {
        baseRisk: 0.15,
        factors: ['geologicalStability', 'historicalActivity', 'faultProximity'],
        timeDecay: 0.95,
        volatility: 0.1
      });
      
      // Flood risk model
      this.riskModels.set('flood', {
        baseRisk: 0.25,
        factors: ['precipitation', 'riverLevels', 'drainageCapacity', 'seasonality'],
        timeDecay: 0.90,
        volatility: 0.2
      });
      
      // Fire weather model
      this.riskModels.set('fire', {
        baseRisk: 0.40,
        factors: ['temperature', 'humidity', 'windSpeed', 'fuelMoisture'],
        timeDecay: 0.85,
        volatility: 0.3
      });
      
      // Severe weather model
      this.riskModels.set('weather', {
        baseRisk: 0.60,
        factors: ['atmosphericPressure', 'temperature', 'humidity', 'windShear'],
        timeDecay: 0.80,
        volatility: 0.25
      });
    }
    
    initializeContextFactors() {
      this.contextFactors.set('populationDensity', {
        low: 0.8,
        medium: 1.0,
        high: 1.3,
        'very-high': 1.6
      });
      
      this.contextFactors.set('infrastructureAge', {
        new: 0.7,
        moderate: 1.0,
        aging: 1.4,
        legacy: 1.8
      });
      
      this.contextFactors.set('seasonality', {
        spring: { flood: 1.3, fire: 0.8, weather: 1.2 },
        summer: { flood: 0.7, fire: 1.5, weather: 1.4 },
        fall: { flood: 1.1, fire: 1.2, weather: 1.1 },
        winter: { flood: 0.9, fire: 0.5, weather: 0.9 }
      });
    }
    
    calculateRiskProbability(hazardType, timeHorizon = '24h', contextOverrides = {}) {
      const model = this.riskModels.get(hazardType);
      if (!model) return 0;
      
      let probability = model.baseRisk;
      
      // Apply time decay
      const hours = this.parseTimeHorizon(timeHorizon);
      probability *= Math.pow(model.timeDecay, hours / 24);
      
      // Apply context factors
      probability *= this.getContextMultiplier(hazardType, contextOverrides);
      
      // Add some realistic volatility
      probability += (Math.random() - 0.5) * model.volatility;
      
      // Clamp between 0 and 1
      return Math.max(0, Math.min(1, probability));
    }
    
    parseTimeHorizon(timeHorizon) {
      const timeMap = {
        '6h': 6,
        '24h': 24,
        '72h': 72,
        '7d': 168
      };
      return timeMap[timeHorizon] || 24;
    }
    
    getContextMultiplier(hazardType, contextOverrides) {
      let multiplier = 1.0;
      
      // Population density factor
      const population = contextOverrides.populationDensity || 'medium';
      multiplier *= this.contextFactors.get('populationDensity')[population] || 1.0;
      
      // Infrastructure age factor
      const infrastructure = contextOverrides.infrastructureAge || 'moderate';
      multiplier *= this.contextFactors.get('infrastructureAge')[infrastructure] || 1.0;
      
      // Seasonal factor
      const season = contextOverrides.season || 'summer';
      const seasonalFactors = this.contextFactors.get('seasonality')[season];
      if (seasonalFactors && seasonalFactors[hazardType]) {
        multiplier *= seasonalFactors[hazardType];
      }
      
      return multiplier;
    }
    
    generateForecast(location, timeHorizon = '24h') {
      const forecast = {
        location,
        timeHorizon,
        timestamp: new Date().toISOString(),
        overallRisk: 0,
        hazards: [],
        confidence: 'medium',
        recommendations: []
      };
      
      // Calculate risk for each hazard type
      for (const [hazardType, model] of this.riskModels) {
        const probability = this.calculateRiskProbability(hazardType, timeHorizon);
        
        forecast.hazards.push({
          type: hazardType,
          probability: Math.round(probability * 100),
          severity: this.calculateSeverity(probability),
          peakTime: this.estimatePeakTime(hazardType, timeHorizon),
          confidence: this.calculateConfidence(hazardType, probability)
        });
        
        // Update overall risk (weighted average)
        forecast.overallRisk += probability * this.getHazardWeight(hazardType);
      }
      
      // Normalize overall risk to 0-10 scale
      forecast.overallRisk = Math.round(forecast.overallRisk * 10 * 100) / 100;
      
      // Generate recommendations
      forecast.recommendations = this.generateRecommendations(forecast);
      
      // Calculate overall confidence
      forecast.confidence = this.calculateOverallConfidence(forecast.hazards);
      
      return forecast;
    }
    
    calculateSeverity(probability) {
      if (probability < 0.3) return 'low';
      if (probability < 0.5) return 'medium';
      if (probability < 0.7) return 'high';
      return 'critical';
    }
    
    estimatePeakTime(hazardType, timeHorizon) {
      const timeMap = {
        earthquake: 'Random occurrence',
        flood: '6-12 hours from now',
        fire: '2-4 PM (peak heat)',
        weather: '2-6 PM today'
      };
      return timeMap[hazardType] || 'Variable';
    }
    
    calculateConfidence(hazardType, probability) {
      // Confidence based on data availability and model accuracy
      const confidenceMap = {
        earthquake: 'medium', // Seismic is inherently unpredictable
        flood: 'high',        // Good precipitation/river data
        fire: 'high',         // Good weather data
        weather: 'high'       // Excellent meteorological data
      };
      
      let confidence = confidenceMap[hazardType] || 'medium';
      
      // Reduce confidence for extreme probabilities (model uncertainty)
      if (probability < 0.1 || probability > 0.9) {
        confidence = confidence === 'high' ? 'medium' : 'low';
      }
      
      return confidence;
    }
    
    getHazardWeight(hazardType) {
      // Weight based on typical impact severity
      const weights = {
        earthquake: 0.3,
        flood: 0.25,
        fire: 0.2,
        weather: 0.25
      };
      return weights[hazardType] || 0.25;
    }
    
    generateRecommendations(forecast) {
      const recommendations = [];
      const highRiskHazards = forecast.hazards.filter(h => h.probability > 50);
      
      if (highRiskHazards.length === 0) {
        recommendations.push({
          priority: 'low',
          timeframe: 'routine',
          action: 'Continue normal monitoring and preparedness activities'
        });
        return recommendations;
      }
      
      // Generate specific recommendations based on high-risk hazards
      highRiskHazards.forEach(hazard => {
        switch (hazard.type) {
          case 'earthquake':
            recommendations.push({
              priority: 'high',
              timeframe: 'immediate',
              action: 'Review earthquake response plans, check emergency supplies'
            });
            break;
          case 'flood':
            recommendations.push({
              priority: 'high',
              timeframe: 'next-6-hours',
              action: 'Monitor water levels, prepare sandbags, alert low-lying areas'
            });
            break;
          case 'fire':
            recommendations.push({
              priority: 'high',
              timeframe: 'next-4-hours',
              action: 'Red flag warning, restrict outdoor burning, pre-position fire crews'
            });
            break;
          case 'weather':
            recommendations.push({
              priority: 'high',
              timeframe: 'next-2-hours',
              action: 'Issue severe weather warning, activate emergency operations'
            });
            break;
        }
      });
      
      return recommendations;
    }
    
    calculateOverallConfidence(hazards) {
      const confidenceValues = { low: 1, medium: 2, high: 3 };
      const avgConfidence = hazards.reduce((sum, h) => sum + confidenceValues[h.confidence], 0) / hazards.length;
      
      if (avgConfidence < 1.5) return 'low';
      if (avgConfidence < 2.5) return 'medium';
      return 'high';
    }
    
    runScenarioAnalysis(scenarios, contextOverrides = {}) {
      const analysis = {
        scenarios: scenarios,
        timestamp: new Date().toISOString(),
        cascadingEffects: [],
        combinedRisk: 0,
        criticalFactors: [],
        mitigationStrategies: []
      };
      
      // Analyze each scenario
      scenarios.forEach(scenario => {
        const scenarioRisk = this.calculateRiskProbability(scenario, '24h', contextOverrides);
        analysis.combinedRisk += scenarioRisk;
        
        // Check for cascading effects
        const cascading = this.analyzeCascadingEffects(scenario, scenarios);
        if (cascading.length > 0) {
          analysis.cascadingEffects.push(...cascading);
        }
      });
      
      // Normalize combined risk
      analysis.combinedRisk = Math.min(1, analysis.combinedRisk);
      
      // Identify critical factors
      analysis.criticalFactors = this.identifyCriticalFactors(scenarios, contextOverrides);
      
      // Generate mitigation strategies
      analysis.mitigationStrategies = this.generateMitigationStrategies(scenarios, analysis.combinedRisk);
      
      return analysis;
    }
    
    analyzeCascadingEffects(primaryScenario, allScenarios) {
      const cascadingMap = {
        earthquake: ['fire', 'flood'], // Earthquakes can trigger fires and dam failures
        flood: ['cyber'],              // Flooding can damage infrastructure
        fire: ['weather'],             // Fires can create their own weather
        cyber: ['weather'],            // Cyber attacks on weather systems
        hurricane: ['flood', 'fire'],  // Hurricanes cause flooding and power issues
        pandemic: ['cyber']            // Pandemic stress on digital infrastructure
      };
      
      const cascading = [];
      const effects = cascadingMap[primaryScenario] || [];
      
      effects.forEach(effect => {
        if (!allScenarios.includes(effect)) {
          cascading.push({
            primary: primaryScenario,
            secondary: effect,
            probability: 0.3 + Math.random() * 0.4, // 30-70% chance
            delay: this.estimateCascadeDelay(primaryScenario, effect)
          });
        }
      });
      
      return cascading;
    }
    
    estimateCascadeDelay(primary, secondary) {
      const delayMap = {
        'earthquake-fire': '15-30 minutes',
        'earthquake-flood': '2-6 hours',
        'flood-cyber': '6-24 hours',
        'fire-weather': '30-60 minutes',
        'hurricane-flood': 'Concurrent',
        'hurricane-fire': '12-48 hours'
      };
      
      return delayMap[`${primary}-${secondary}`] || '1-6 hours';
    }
    
    identifyCriticalFactors(scenarios, contextOverrides) {
      const factors = [];
      
      // High population density + multiple scenarios = critical
      if (scenarios.length > 1 && contextOverrides.populationDensity === 'high') {
        factors.push('High population density amplifies multi-hazard impact');
      }
      
      // Aging infrastructure + physical hazards = critical
      if ((scenarios.includes('earthquake') || scenarios.includes('flood')) && 
          contextOverrides.infrastructureAge === 'aging') {
        factors.push('Aging infrastructure vulnerable to physical hazards');
      }
      
      // Cyber + physical = critical
      if (scenarios.includes('cyber') && scenarios.some(s => ['earthquake', 'flood', 'fire', 'hurricane'].includes(s))) {
        factors.push('Cyber-physical attack combination creates systemic risk');
      }
      
      return factors;
    }
    
    generateMitigationStrategies(scenarios, combinedRisk) {
      const strategies = [];
      
      if (combinedRisk > 0.7) {
        strategies.push({
          priority: 'critical',
          action: 'Activate emergency operations center immediately',
          timeframe: 'Next 30 minutes'
        });
        
        strategies.push({
          priority: 'critical',
          action: 'Issue public emergency alert with evacuation guidance',
          timeframe: 'Next 1 hour'
        });
      }
      
      if (scenarios.length > 2) {
        strategies.push({
          priority: 'high',
          action: 'Coordinate multi-agency response due to complex scenario',
          timeframe: 'Next 2 hours'
        });
      }
      
      // Scenario-specific strategies
      if (scenarios.includes('cyber')) {
        strategies.push({
          priority: 'high',
          action: 'Activate backup communication systems and isolate critical networks',
          timeframe: 'Immediate'
        });
      }
      
      if (scenarios.includes('pandemic')) {
        strategies.push({
          priority: 'medium',
          action: 'Implement social distancing protocols in emergency operations',
          timeframe: 'Next 4 hours'
        });
      }
      
      return strategies;
    }
    
    startRealTimeUpdates(callback) {
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
      }
      
      this.updateInterval = setInterval(() => {
        // Simulate real-time data updates
        const updates = this.generateRealTimeUpdates();
        callback(updates);
      }, 5000);
    }
    
    stopRealTimeUpdates() {
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
        this.updateInterval = null;
      }
    }
    
    generateRealTimeUpdates() {
      const updateTypes = [
        'Weather station data',
        'Seismic sensor network',
        'Emergency services status',
        'Traffic monitoring',
        'Communication networks',
        'Utility grid status',
        'Public alert system',
        'Resource allocation'
      ];
      
      const statuses = [
        'reports normal conditions',
        'shows elevated readings',
        'indicates minor anomaly',
        'confirms operational status',
        'detects pattern change',
        'updates baseline metrics',
        'completes status check',
        'processes new data'
      ];
      
      const randomType = updateTypes[Math.floor(Math.random() * updateTypes.length)];
      const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];
      
      return {
        timestamp: new Date().toLocaleTimeString(),
        source: randomType,
        message: `${randomType} ${randomStatus}`,
        severity: Math.random() < 0.1 ? 'high' : Math.random() < 0.3 ? 'medium' : 'low'
      };
    }
  }
  
  // Global instance
  window.PredictiveRisk = new PredictiveRiskEngine();
  
  // Integration functions for the UI
  function initializePredictiveRisk() {
    console.log('🔮 Initializing Predictive Risk UI...');
    
    // Start real-time updates
    window.PredictiveRisk.startRealTimeUpdates((update) => {
      addRealTimeUpdate(update);
    });
    
    // Load initial forecast
    updateForecastFromEngine();
  }
  
  function updateForecastFromEngine() {
    const contextOverrides = {
      populationDensity: document.getElementById('population')?.value || 'medium',
      infrastructureAge: document.getElementById('infrastructure')?.value || 'moderate',
      season: document.getElementById('season')?.value || 'summer'
    };
    
    const location = document.getElementById('location')?.value || 'Unknown Location';
    const forecast = window.PredictiveRisk.generateForecast(location, '24h');
    
    // Update UI with forecast data
    updateRiskDisplay(forecast);
    updateHazardCards(forecast.hazards);
  }
  
  function updateRiskDisplay(forecast) {
    const riskElement = document.getElementById('overallRiskLevel');
    if (riskElement) {
      riskElement.textContent = forecast.overallRisk;
      
      // Update risk color
      riskElement.className = 'risk-level ';
      if (forecast.overallRisk < 3) riskElement.className += 'risk-low';
      else if (forecast.overallRisk < 5) riskElement.className += 'risk-medium';
      else if (forecast.overallRisk < 7) riskElement.className += 'risk-high';
      else riskElement.className += 'risk-critical';
    }
  }
  
  function updateHazardCards(hazards) {
    hazards.forEach(hazard => {
      const card = document.querySelector(`.hazard-card.${hazard.type}`);
      if (card) {
        const fill = card.querySelector('.probability-fill');
        const text = card.querySelector('.probability-text');
        
        if (fill) fill.style.width = hazard.probability + '%';
        if (text) text.textContent = `${hazard.probability}% probability in next 24h`;
      }
    });
  }
  
  function runScenarioAnalysisFromUI() {
    const selectedScenarios = Array.from(document.querySelectorAll('.scenario-option.selected'))
      .map(el => el.dataset.scenario);
    
    if (selectedScenarios.length === 0) return;
    
    const contextOverrides = {
      populationDensity: document.getElementById('population')?.value || 'medium',
      infrastructureAge: document.getElementById('infrastructure')?.value || 'moderate',
      season: document.getElementById('season')?.value || 'summer'
    };
    
    const analysis = window.PredictiveRisk.runScenarioAnalysis(selectedScenarios, contextOverrides);
    
    // Update analysis results display
    updateAnalysisResults(analysis);
  }
  
  function updateAnalysisResults(analysis) {
    const resultsDiv = document.getElementById('analysisResults');
    if (!resultsDiv) return;
    
    // Update the analysis content with real data
    const summaryDiv = resultsDiv.querySelector('div > p');
    if (summaryDiv) {
      summaryDiv.innerHTML = `
        Based on scenario analysis of <strong>${analysis.scenarios.join(', ')}</strong>, 
        the combined risk level is <strong>${Math.round(analysis.combinedRisk * 100)}%</strong>. 
        ${analysis.cascadingEffects.length > 0 ? 'Cascading effects detected.' : 'No major cascading effects identified.'}
      `;
    }
    
    resultsDiv.classList.add('visible');
  }
  
  function addRealTimeUpdate(update) {
    const stream = document.getElementById('updateStream');
    if (!stream) return;
    
    const newItem = document.createElement('div');
    newItem.className = 'update-item';
    newItem.innerHTML = `
      <span class="update-timestamp">${update.timestamp}</span> - ${update.message}
    `;
    
    // Add severity indicator if high
    if (update.severity === 'high') {
      newItem.style.background = 'rgba(239, 68, 68, 0.1)';
      newItem.style.borderLeft = '3px solid #ef4444';
    }
    
    stream.insertBefore(newItem, stream.firstChild);
    
    // Keep only last 10 updates
    while (stream.children.length > 10) {
      stream.removeChild(stream.lastChild);
    }
  }
  
  // Export for other modules
  export { PredictiveRiskEngine, initializePredictiveRisk };