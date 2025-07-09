// Real-Time Resource Optimizer Engine
// AI-powered dynamic resource allocation and deployment optimization

class ResourceOptimizerEngine {
    constructor() {
      this.resources = new Map();
      this.deploymentQueue = [];
      this.optimizationHistory = [];
      this.realTimeData = new Map();
      this.performanceMetrics = new Map();
      
      // Initialize resource types and their properties
      this.initializeResourceTypes();
      this.initializePerformanceMetrics();
      this.loadResourceInventory();
      
      console.log('⚡ Resource Optimizer Engine initialized');
    }
    
    initializeResourceTypes() {
      this.resourceTypes = new Map([
        ['fire', {
          icon: '🚒',
          name: 'Fire Suppression',
          baseResponseTime: 5.0,
          coverageRadius: 3.5,
          operationalCost: 850,
          capacityFactor: 1.2,
          specializations: ['structure-fire', 'wildfire', 'hazmat'],
          minimumCrew: 3,
          maxDeploymentTime: 24
        }],
        ['medical', {
          icon: '🚑',
          name: 'Emergency Medical',
          baseResponseTime: 6.5,
          coverageRadius: 4.0,
          operationalCost: 720,
          capacityFactor: 1.0,
          specializations: ['trauma', 'cardiac', 'pediatric'],
          minimumCrew: 2,
          maxDeploymentTime: 16
        }],
        ['police', {
          icon: '🚔',
          name: 'Law Enforcement',
          baseResponseTime: 4.2,
          coverageRadius: 5.0,
          operationalCost: 650,
          capacityFactor: 0.8,
          specializations: ['patrol', 'swat', 'traffic'],
          minimumCrew: 1,
          maxDeploymentTime: 12
        }],
        ['rescue', {
          icon: '🚁',
          name: 'Search & Rescue',
          baseResponseTime: 15.0,
          coverageRadius: 25.0,
          operationalCost: 2400,
          capacityFactor: 2.5,
          specializations: ['mountain', 'water', 'urban'],
          minimumCrew: 4,
          maxDeploymentTime: 48
        }],
        ['logistics', {
          icon: '🚛',
          name: 'Logistics Support',
          baseResponseTime: 20.0,
          coverageRadius: 8.0,
          operationalCost: 450,
          capacityFactor: 1.8,
          specializations: ['supply', 'transport', 'staging'],
          minimumCrew: 2,
          maxDeploymentTime: 72
        }],
        ['hazmat', {
          icon: '🛡️',
          name: 'Hazmat Teams',
          baseResponseTime: 25.0,
          coverageRadius: 6.0,
          operationalCost: 1800,
          capacityFactor: 3.0,
          specializations: ['chemical', 'biological', 'nuclear'],
          minimumCrew: 6,
          maxDeploymentTime: 36
        }]
      ]);
    }
    
    initializePerformanceMetrics() {
      this.performanceMetrics.set('totalUnits', 247);
      this.performanceMetrics.set('deployedUnits', 89);
      this.performanceMetrics.set('availableUnits', 158);
      this.performanceMetrics.set('maintenanceUnits', 12);
      this.performanceMetrics.set('overallEfficiency', 94);
      this.performanceMetrics.set('averageResponseTime', 6.8);
      this.performanceMetrics.set('coveragePercentage', 87);
      this.performanceMetrics.set('costPerOperation', 1250);
    }
    
    loadResourceInventory() {
      // Initialize resource inventory for each type
      for (const [type, config] of this.resourceTypes) {
        const unitCount = this.calculateOptimalUnitCount(type);
        const units = [];
        
        for (let i = 1; i <= unitCount; i++) {
          units.push({
            id: `${type}-${i}`,
            type: type,
            status: this.getRandomStatus(),
            location: this.getRandomLocation(),
            lastDeployment: this.getRandomLastDeployment(),
            crew: Math.ceil(Math.random() * config.minimumCrew * 2),
            efficiency: 75 + Math.random() * 25,
            maintenanceScore: 80 + Math.random() * 20,
            specializations: this.getRandomSpecializations(config.specializations)
          });
        }
        
        this.resources.set(type, units);
      }
    }
    
    calculateOptimalUnitCount(type) {
      const baseCounts = {
        fire: 15,
        medical: 22,
        police: 28,
        rescue: 8,
        logistics: 18,
        hazmat: 6
      };
      
      const baseCount = baseCounts[type] || 10;
      const variation = Math.round((Math.random() - 0.5) * 6);
      return Math.max(5, baseCount + variation);
    }
    
    getRandomStatus() {
      const statuses = ['available', 'deployed', 'engaged', 'maintenance'];
      const weights = [0.5, 0.3, 0.15, 0.05]; // Probability weights
      
      const random = Math.random();
      let cumulative = 0;
      
      for (let i = 0; i < statuses.length; i++) {
        cumulative += weights[i];
        if (random < cumulative) return statuses[i];
      }
      
      return 'available';
    }
    
    getRandomLocation() {
      const sectors = ['1-A', '1-B', '2-A', '2-B', '3-A', '3-B', '4-A', '4-B', '5-A', '5-B'];
      return `Sector ${sectors[Math.floor(Math.random() * sectors.length)]}`;
    }
    
    getRandomLastDeployment() {
      const hours = Math.random() * 48; // Last 48 hours
      const deploymentTime = new Date(Date.now() - hours * 60 * 60 * 1000);
      return deploymentTime.toISOString();
    }
    
    getRandomSpecializations(availableSpecs) {
      const count = Math.floor(Math.random() * availableSpecs.length) + 1;
      const shuffled = [...availableSpecs].sort(() => Math.random() - 0.5);
      return shuffled.slice(0, count);
    }
    
    optimizeResourceAllocation(parameters = {}) {
      const {
        priorityMode = 'coverage',
        deploymentStrategy = 'reactive',
        riskTolerance = 'moderate',
        timeHorizon = '4h',
        emergencyLevel = 'normal'
      } = parameters;
      
      console.log('🧠 Starting resource optimization...', parameters);
      
      const optimization = {
        timestamp: new Date().toISOString(),
        parameters,
        currentState: this.analyzeCurrentState(),
        recommendations: [],
        projectedImprovements: {},
        deploymentPlan: [],
        costAnalysis: {}
      };
      
      // Analyze current resource distribution
      const currentAnalysis = this.analyzeResourceDistribution();
      optimization.currentAnalysis = currentAnalysis;
      
      // Generate optimization recommendations
      optimization.recommendations = this.generateOptimizationRecommendations(
        currentAnalysis, parameters
      );
      
      // Calculate projected improvements
      optimization.projectedImprovements = this.calculateProjectedImprovements(
        optimization.recommendations
      );
      
      // Create deployment plan
      optimization.deploymentPlan = this.createDeploymentPlan(
        optimization.recommendations, parameters
      );
      
      // Analyze cost implications
      optimization.costAnalysis = this.analyzeCostImplications(optimization.deploymentPlan);
      
      // Store in optimization history
      this.optimizationHistory.push(optimization);
      
      return optimization;
    }
    
    analyzeCurrentState() {
      const state = {
        totalResources: 0,
        deployedResources: 0,
        availableResources: 0,
        maintenanceResources: 0,
        coverageGaps: [],
        overallEfficiency: 0,
        responseTimeAverage: 0
      };
      
      let totalEfficiency = 0;
      let totalResponseTime = 0;
      let unitCount = 0;
      
      for (const [type, units] of this.resources) {
        const typeConfig = this.resourceTypes.get(type);
        
        for (const unit of units) {
          state.totalResources++;
          unitCount++;
          
          switch (unit.status) {
            case 'deployed':
            case 'engaged':
              state.deployedResources++;
              break;
            case 'available':
              state.availableResources++;
              break;
            case 'maintenance':
              state.maintenanceResources++;
              break;
          }
          
          totalEfficiency += unit.efficiency;
          totalResponseTime += typeConfig.baseResponseTime * (2 - unit.efficiency / 100);
        }
      }
      
      state.overallEfficiency = totalEfficiency / unitCount;
      state.responseTimeAverage = totalResponseTime / unitCount;
      
      return state;
    }
    
    analyzeResourceDistribution() {
      const analysis = {
        byType: new Map(),
        bySector: new Map(),
        coverageAnalysis: {},
        bottlenecks: [],
        recommendations: []
      };
      
      // Analyze by resource type
      for (const [type, units] of this.resources) {
        const typeAnalysis = {
          total: units.length,
          available: units.filter(u => u.status === 'available').length,
          deployed: units.filter(u => u.status === 'deployed' || u.status === 'engaged').length,
          maintenance: units.filter(u => u.status === 'maintenance').length,
          averageEfficiency: units.reduce((sum, u) => sum + u.efficiency, 0) / units.length,
          coverageScore: this.calculateCoverageScore(type, units)
        };
        
        analysis.byType.set(type, typeAnalysis);
        
        // Identify bottlenecks
        if (typeAnalysis.available < typeAnalysis.total * 0.3) {
          analysis.bottlenecks.push({
            type: 'low-availability',
            resourceType: type,
            severity: 'high',
            description: `Low availability for ${type} units (${typeAnalysis.available}/${typeAnalysis.total})`
          });
        }
      }
      
      return analysis;
    }
    
    calculateCoverageScore(type, units) {
      const config = this.resourceTypes.get(type);
      const availableUnits = units.filter(u => u.status === 'available' || u.status === 'deployed');
      
      // Simplified coverage calculation based on unit count and radius
      const theoreticalCoverage = availableUnits.length * config.coverageRadius * config.coverageRadius * Math.PI;
      const totalArea = 500; // Assume 500 sq km service area
      
      return Math.min(100, (theoreticalCoverage / totalArea) * 100);
    }
    
    generateOptimizationRecommendations(analysis, parameters) {
      const recommendations = [];
      
      // Analyze each resource type for optimization opportunities
      for (const [type, typeAnalysis] of analysis.byType) {
        const config = this.resourceTypes.get(type);
        
        // Low availability recommendations
        if (typeAnalysis.available / typeAnalysis.total < 0.4) {
          recommendations.push({
            priority: 'high',
            type: 'reallocation',
            resourceType: type,
            action: 'recall-units',
            description: `Recall non-critical ${type} units to improve availability`,
            impact: {
              responseTime: -15,
              coverage: +10,
              cost: -5
            }
          });
        }
        
        // Coverage gap recommendations
        if (typeAnalysis.coverageScore < 70) {
          recommendations.push({
            priority: 'medium',
            type: 'positioning',
            resourceType: type,
            action: 'reposition-units',
            description: `Reposition ${type} units to improve coverage`,
            impact: {
              responseTime: -10,
              coverage: +20,
              cost: +2
            }
          });
        }
        
        // Efficiency recommendations
        if (typeAnalysis.averageEfficiency < 85) {
          recommendations.push({
            priority: 'low',
            type: 'maintenance',
            resourceType: type,
            action: 'schedule-maintenance',
            description: `Schedule maintenance for ${type} units to improve efficiency`,
            impact: {
              efficiency: +15,
              reliability: +25,
              cost: +8
            }
          });
        }
      }
      
      // Strategic recommendations based on priority mode
      switch (parameters.priorityMode) {
        case 'response-time':
          recommendations.push({
            priority: 'high',
            type: 'strategy',
            action: 'forward-deploy',
            description: 'Forward deploy units to high-demand areas',
            impact: { responseTime: -25, coverage: +5, cost: +15 }
          });
          break;
          
        case 'coverage':
          recommendations.push({
            priority: 'high',
            type: 'strategy',
            action: 'distribute-evenly',
            description: 'Distribute units evenly across service area',
            impact: { coverage: +30, responseTime: +5, cost: -5 }
          });
          break;
          
        case 'risk-based':
          recommendations.push({
            priority: 'high',
            type: 'strategy',
            action: 'risk-weighted-deploy',
            description: 'Deploy units based on risk assessment',
            impact: { effectiveness: +40, responseTime: -10, cost: +10 }
          });
          break;
      }
      
      return recommendations.sort((a, b) => {
        const priorityOrder = { high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      });
    }
    
    calculateProjectedImprovements(recommendations) {
      let improvements = {
        responseTime: 0,
        coverage: 0,
        efficiency: 0,
        cost: 0,
        overall: 0
      };
      
      // Calculate cumulative impact of all recommendations
      for (const rec of recommendations) {
        if (rec.impact) {
          improvements.responseTime += rec.impact.responseTime || 0;
          improvements.coverage += rec.impact.coverage || 0;
          improvements.efficiency += rec.impact.efficiency || 0;
          improvements.cost += rec.impact.cost || 0;
        }
      }
      
      // Calculate overall improvement score
      improvements.overall = (
        Math.abs(improvements.responseTime) * 0.3 +
        improvements.coverage * 0.3 +
        improvements.efficiency * 0.2 +
        Math.abs(improvements.cost) * 0.2
      );
      
      return improvements;
    }
    
    createDeploymentPlan(recommendations, parameters) {
      const plan = {
        immediate: [], // Next 30 minutes
        shortTerm: [], // Next 2 hours
        mediumTerm: [], // Next 8 hours
        longTerm: [] // Next 24+ hours
      };
      
      recommendations.forEach(rec => {
        const deployment = {
          recommendation: rec,
          units: this.identifyUnitsForRecommendation(rec),
          timeline: this.estimateImplementationTime(rec),
          resources: this.calculateRequiredResources(rec)
        };
        
        // Categorize by timeline
        if (deployment.timeline <= 30) {
          plan.immediate.push(deployment);
        } else if (deployment.timeline <= 120) {
          plan.shortTerm.push(deployment);
        } else if (deployment.timeline <= 480) {
          plan.mediumTerm.push(deployment);
        } else {
          plan.longTerm.push(deployment);
        }
      });
      
      return plan;
    }
    
    identifyUnitsForRecommendation(recommendation) {
      const relevantUnits = [];
      
      if (recommendation.resourceType) {
        const units = this.resources.get(recommendation.resourceType) || [];
        
        switch (recommendation.action) {
          case 'recall-units':
            relevantUnits.push(...units.filter(u => u.status === 'deployed' || u.status === 'engaged').slice(0, 3));
            break;
          case 'reposition-units':
            relevantUnits.push(...units.filter(u => u.status === 'available').slice(0, 2));
            break;
          case 'schedule-maintenance':
            relevantUnits.push(...units.filter(u => u.maintenanceScore < 85).slice(0, 1));
            break;
        }
      }
      
      return relevantUnits;
    }
    
    estimateImplementationTime(recommendation) {
      const timeEstimates = {
        'recall-units': 15,
        'reposition-units': 45,
        'forward-deploy': 30,
        'distribute-evenly': 120,
        'risk-weighted-deploy': 90,
        'schedule-maintenance': 480
      };
      
      return timeEstimates[recommendation.action] || 60;
    }
    
    calculateRequiredResources(recommendation) {
      return {
        personnel: Math.ceil(Math.random() * 10) + 2,
        vehicles: Math.ceil(Math.random() * 3) + 1,
        equipment: Math.ceil(Math.random() * 5) + 1,
        estimatedCost: Math.ceil(Math.random() * 50000) + 10000
      };
    }
    
    analyzeCostImplications(deploymentPlan) {
      const analysis = {
        immediate: 0,
        shortTerm: 0,
        mediumTerm: 0,
        longTerm: 0,
        total: 0,
        breakdown: {
          personnel: 0,
          fuel: 0,
          equipment: 0,
          maintenance: 0,
          other: 0
        }
      };
      
      // Calculate costs for each timeline
      Object.entries(deploymentPlan).forEach(([timeline, deployments]) => {
        const timelineCost = deployments.reduce((sum, dep) => {
          return sum + (dep.resources?.estimatedCost || 0);
        }, 0);
        
        analysis[timeline] = timelineCost;
        analysis.total += timelineCost;
      });
      
      // Estimate cost breakdown
      analysis.breakdown.personnel = analysis.total * 0.45;
      analysis.breakdown.fuel = analysis.total * 0.25;
      analysis.breakdown.equipment = analysis.total * 0.15;
      analysis.breakdown.maintenance = analysis.total * 0.10;
      analysis.breakdown.other = analysis.total * 0.05;
      
      return analysis;
    }
    
    simulateDeployment(unitId, targetLocation, duration = 60) {
      // Find the unit
      let unit = null;
      let resourceType = null;
      
      for (const [type, units] of this.resources) {
        const foundUnit = units.find(u => u.id === unitId);
        if (foundUnit) {
          unit = foundUnit;
          resourceType = type;
          break;
        }
      }
      
      if (!unit) {
        throw new Error(`Unit ${unitId} not found`);
      }
      
      if (unit.status !== 'available') {
        throw new Error(`Unit ${unitId} is not available (status: ${unit.status})`);
      }
      
      // Create deployment record
      const deployment = {
        id: `dep-${Date.now()}`,
        unitId,
        resourceType,
        targetLocation,
        startTime: new Date().toISOString(),
        estimatedDuration: duration,
        status: 'in-progress',
        progress: 0
      };
      
      // Update unit status
      unit.status = 'deployed';
      unit.location = targetLocation;
      
      // Add to deployment queue
      this.deploymentQueue.push(deployment);
      
      // Simulate deployment progress
      this.simulateDeploymentProgress(deployment);
      
      return deployment;
    }
    
    simulateDeploymentProgress(deployment) {
      const progressInterval = setInterval(() => {
        deployment.progress += Math.random() * 20 + 5; // 5-25% progress per update
        
        if (deployment.progress >= 100) {
          deployment.progress = 100;
          deployment.status = 'complete';
          deployment.endTime = new Date().toISOString();
          
          // Find and update unit status back to available
          for (const [type, units] of this.resources) {
            const unit = units.find(u => u.id === deployment.unitId);
            if (unit) {
              unit.status = 'available';
              unit.lastDeployment = deployment.endTime;
              break;
            }
          }
          
          // Remove from deployment queue
          const index = this.deploymentQueue.findIndex(d => d.id === deployment.id);
          if (index !== -1) {
            this.deploymentQueue.splice(index, 1);
          }
          
          clearInterval(progressInterval);
          console.log(`✅ Deployment ${deployment.id} completed`);
        }
      }, 2000); // Update every 2 seconds
    }
    
    getResourceSummary() {
      const summary = {
        total: 0,
        byType: new Map(),
        byStatus: new Map(),
        efficiency: 0,
        coverage: 0
      };
      
      let totalEfficiency = 0;
      let totalCoverage = 0;
      
      for (const [type, units] of this.resources) {
        const typeConfig = this.resourceTypes.get(type);
        const typeSummary = {
          total: units.length,
          available: 0,
          deployed: 0,
          engaged: 0,
          maintenance: 0,
          efficiency: 0,
          coverage: 0
        };
        
        let typeEfficiency = 0;
        
        units.forEach(unit => {
          summary.total++;
          typeSummary[unit.status]++;
          typeEfficiency += unit.efficiency;
        });
        
        typeSummary.efficiency = typeEfficiency / units.length;
        typeSummary.coverage = this.calculateCoverageScore(type, units);
        
        summary.byType.set(type, typeSummary);
        
        totalEfficiency += typeSummary.efficiency * units.length;
        totalCoverage += typeSummary.coverage;
      }
      
      summary.efficiency = totalEfficiency / summary.total;
      summary.coverage = totalCoverage / this.resourceTypes.size;
      
      // Calculate status totals
      for (const [type, typeSummary] of summary.byType) {
        ['available', 'deployed', 'engaged', 'maintenance'].forEach(status => {
          summary.byStatus.set(status, (summary.byStatus.get(status) || 0) + typeSummary[status]);
        });
      }
      
      return summary;
    }
    
    predictResourceDemand(timeHorizon = '4h', contextData = {}) {
      const prediction = {
        timeHorizon,
        timestamp: new Date().toISOString(),
        demandByType: new Map(),
        totalDemand: 0,
        peakPeriods: [],
        recommendations: []
      };
      
      const hours = this.parseTimeHorizon(timeHorizon);
      
      // Base demand patterns (calls per hour)
      const baseDemand = {
        fire: 0.8,
        medical: 2.4,
        police: 3.6,
        rescue: 0.3,
        logistics: 0.5,
        hazmat: 0.1
      };
      
      // Apply time-based modifiers
      const currentHour = new Date().getHours();
      const timeModifiers = this.getTimeBasedModifiers(currentHour);
      
      // Apply context modifiers
      const contextModifiers = this.getContextModifiers(contextData);
      
      // Calculate predicted demand for each resource type
      for (const [type, baseRate] of Object.entries(baseDemand)) {
        const timeModifier = timeModifiers[type] || 1.0;
        const contextModifier = contextModifiers[type] || 1.0;
        
        const adjustedRate = baseRate * timeModifier * contextModifier;
        const totalDemand = adjustedRate * hours;
        
        prediction.demandByType.set(type, {
          baseRate,
          adjustedRate,
          totalDemand: Math.round(totalDemand * 10) / 10,
          peakHours: this.identifyPeakHours(type, hours),
          confidence: this.calculateDemandConfidence(type, contextData)
        });
        
        prediction.totalDemand += totalDemand;
      }
      
      // Identify overall peak periods
      prediction.peakPeriods = this.identifyOverallPeakPeriods(prediction.demandByType, hours);
      
      // Generate recommendations based on predicted demand
      prediction.recommendations = this.generateDemandRecommendations(prediction);
      
      return prediction;
    }
    
    parseTimeHorizon(timeHorizon) {
      const timeMap = {
        '1h': 1,
        '4h': 4,
        '12h': 12,
        '24h': 24
      };
      return timeMap[timeHorizon] || 4;
    }
    
    getTimeBasedModifiers(currentHour) {
      // Different resource types have different demand patterns throughout the day
      const modifiers = {
        fire: 1.0,
        medical: 1.0,
        police: 1.0,
        rescue: 1.0,
        logistics: 1.0,
        hazmat: 1.0
      };
      
      // Rush hour increases (7-9 AM, 5-7 PM)
      if ((currentHour >= 7 && currentHour <= 9) || (currentHour >= 17 && currentHour <= 19)) {
        modifiers.police *= 1.8;
        modifiers.medical *= 1.4;
        modifiers.fire *= 1.2;
      }
      
      // Evening/night patterns (10 PM - 2 AM)
      if (currentHour >= 22 || currentHour <= 2) {
        modifiers.police *= 1.6;
        modifiers.medical *= 1.3;
        modifiers.fire *= 0.8;
      }
      
      // Business hours (9 AM - 5 PM)
      if (currentHour >= 9 && currentHour <= 17) {
        modifiers.hazmat *= 1.5;
        modifiers.logistics *= 1.3;
      }
      
      return modifiers;
    }
    
    getContextModifiers(contextData) {
      const modifiers = {
        fire: 1.0,
        medical: 1.0,
        police: 1.0,
        rescue: 1.0,
        logistics: 1.0,
        hazmat: 1.0
      };
      
      // Weather-based modifiers
      if (contextData.weather) {
        switch (contextData.weather) {
          case 'severe-storm':
            modifiers.rescue *= 2.5;
            modifiers.fire *= 1.8;
            modifiers.medical *= 1.5;
            break;
          case 'high-wind':
            modifiers.fire *= 2.0;
            modifiers.rescue *= 1.5;
            break;
          case 'heavy-rain':
            modifiers.rescue *= 1.8;
            modifiers.medical *= 1.3;
            break;
        }
      }
      
      // Event-based modifiers
      if (contextData.events) {
        if (contextData.events.includes('large-gathering')) {
          modifiers.medical *= 1.4;
          modifiers.police *= 1.6;
        }
        if (contextData.events.includes('construction')) {
          modifiers.hazmat *= 1.3;
          modifiers.logistics *= 1.2;
        }
      }
      
      // Risk level modifiers
      if (contextData.riskLevel) {
        const riskMultiplier = {
          low: 0.8,
          medium: 1.0,
          high: 1.5,
          critical: 2.0
        }[contextData.riskLevel] || 1.0;
        
        Object.keys(modifiers).forEach(type => {
          modifiers[type] *= riskMultiplier;
        });
      }
      
      return modifiers;
    }
    
    identifyPeakHours(resourceType, totalHours) {
      // Simplified peak hour identification
      const peakPatterns = {
        fire: [14, 15, 16], // 2-4 PM (hottest part of day)
        medical: [10, 11, 18, 19], // Late morning and early evening
        police: [20, 21, 22, 23], // Evening hours
        rescue: [13, 14, 15, 16], // Afternoon outdoor activities
        logistics: [8, 9, 16, 17], // Business transition hours
        hazmat: [10, 11, 14, 15] // Business hours
      };
      
      const peaks = peakPatterns[resourceType] || [];
      const currentHour = new Date().getHours();
      
      return peaks.filter(hour => {
        const hourDiff = (hour - currentHour + 24) % 24;
        return hourDiff <= totalHours;
      });
    }
    
    identifyOverallPeakPeriods(demandByType, totalHours) {
      const periods = [];
      const currentHour = new Date().getHours();
      
      for (let i = 0; i < totalHours; i++) {
        const hour = (currentHour + i) % 24;
        let totalDemandForHour = 0;
        
        for (const [type, data] of demandByType) {
          if (data.peakHours && data.peakHours.includes(hour)) {
            totalDemandForHour += data.adjustedRate * 1.5; // Peak multiplier
          } else {
            totalDemandForHour += data.adjustedRate;
          }
        }
        
        if (totalDemandForHour > 8) { // Threshold for peak period
          periods.push({
            hour,
            demandLevel: totalDemandForHour,
            severity: totalDemandForHour > 12 ? 'high' : 'medium'
          });
        }
      }
      
      return periods;
    }
    
    calculateDemandConfidence(resourceType, contextData) {
      let confidence = 0.7; // Base confidence
      
      // Increase confidence based on available data
      if (contextData.weather) confidence += 0.1;
      if (contextData.events) confidence += 0.1;
      if (contextData.historicalData) confidence += 0.15;
      
      // Resource-specific confidence adjustments
      const typeConfidence = {
        medical: 0.9,  // Medical demand is most predictable
        police: 0.8,   // Police demand fairly predictable
        fire: 0.6,     // Fire demand less predictable
        rescue: 0.5,   // Rescue demand highly variable
        logistics: 0.7,
        hazmat: 0.4    // Hazmat very unpredictable
      };
      
      confidence *= typeConfidence[resourceType] || 0.6;
      
      return Math.min(0.95, Math.max(0.3, confidence));
    }
    
    generateDemandRecommendations(prediction) {
      const recommendations = [];
      
      // Check each resource type for potential issues
      for (const [type, demandData] of prediction.demandByType) {
        const currentResources = this.resources.get(type) || [];
        const availableUnits = currentResources.filter(u => u.status === 'available').length;
        
        // High demand warning
        if (demandData.totalDemand > availableUnits * 0.8) {
          recommendations.push({
            priority: 'high',
            type: 'capacity-warning',
            resourceType: type,
            message: `High demand predicted for ${type} (${demandData.totalDemand} calls vs ${availableUnits} available units)`,
            actions: ['recall-units', 'request-mutual-aid', 'extend-shifts']
          });
        }
        
        // Peak period preparations
        if (demandData.peakHours && demandData.peakHours.length > 0) {
          recommendations.push({
            priority: 'medium',
            type: 'peak-preparation',
            resourceType: type,
            message: `Peak demand expected for ${type} at hours: ${demandData.peakHours.join(', ')}`,
            actions: ['pre-position-units', 'schedule-overtime', 'alert-standby-crews']
          });
        }
      }
      
      // Overall system recommendations
      if (prediction.peakPeriods.length > 0) {
        const highSeverityPeriods = prediction.peakPeriods.filter(p => p.severity === 'high');
        
        if (highSeverityPeriods.length > 0) {
          recommendations.push({
            priority: 'critical',
            type: 'system-wide',
            message: `System-wide high demand periods identified`,
            actions: ['activate-emergency-operations', 'coordinate-mutual-aid', 'implement-surge-protocols']
          });
        }
      }
      
      return recommendations;
    }
    
    generateResourceReport() {
      const summary = this.getResourceSummary();
      const currentState = this.analyzeCurrentState();
      const lastOptimization = this.optimizationHistory[this.optimizationHistory.length - 1];
      
      const report = {
        timestamp: new Date().toISOString(),
        summary,
        currentState,
        optimization: lastOptimization,
        recommendations: [],
        alerts: [],
        performance: {
          efficiency: summary.efficiency,
          coverage: summary.coverage,
          responseTime: currentState.responseTimeAverage,
          availability: (summary.byStatus.get('available') || 0) / summary.total * 100
        }
      };
      
      // Generate alerts for critical issues
      if (report.performance.availability < 40) {
        report.alerts.push({
          level: 'critical',
          message: 'Low resource availability - immediate action required',
          recommendation: 'Recall non-essential deployments and activate reserves'
        });
      }
      
      if (report.performance.efficiency < 80) {
        report.alerts.push({
          level: 'warning',
          message: 'Below-target system efficiency',
          recommendation: 'Schedule maintenance and training activities'
        });
      }
      
      return report;
    }
  }
  
  // Global instance
  window.ResourceOptimizerEngine = new ResourceOptimizerEngine();
  
  // UI Integration Functions
  function initializeResourceOptimizer() {
    console.log('⚡ Initializing Resource Optimizer UI...');
    
    // Load initial data
    updateResourceMetrics();
    loadResourceConfiguration();
    
    // Set up periodic updates
    setInterval(() => {
      updateResourceMetrics();
      updatePerformanceIndicators();
    }, 10000); // Update every 10 seconds
  }
  
  function updateResourceMetrics() {
    const summary = window.ResourceOptimizerEngine.getResourceSummary();
    
    // Update main metrics
    const totalElement = document.getElementById('totalUnits');
    const deployedElement = document.getElementById('deployedUnits');
    const efficiencyElement = document.getElementById('efficiencyScore');
    
    if (totalElement) totalElement.textContent = summary.total;
    if (deployedElement) {
      const deployed = (summary.byStatus.get('deployed') || 0) + (summary.byStatus.get('engaged') || 0);
      deployedElement.textContent = deployed;
    }
    if (efficiencyElement) {
      efficiencyElement.textContent = Math.round(summary.efficiency) + '%';
    }
  }
  
  function updatePerformanceIndicators() {
    // Update unit cards with live data
    const summary = window.ResourceOptimizerEngine.getResourceSummary();
    
    for (const [type, typeData] of summary.byType) {
      const card = document.querySelector(`.unit-card.${type}`);
      if (card) {
        const details = card.querySelector('.unit-details');
        if (details) {
          const children = details.children;
          if (children[0]) children[0].innerHTML = `Units: <strong>${typeData.total} Active</strong>`;
          if (children[1]) children[1].innerHTML = `Coverage: <strong>${Math.round(typeData.coverage)}%</strong>`;
          if (children[3]) children[3].innerHTML = `Efficiency: <strong>${Math.round(typeData.efficiency)}%</strong>`;
        }
      }
    }
  }
  
  function loadResourceConfiguration() {
    const saved = localStorage.getItem('resourceOptimizerConfig');
    if (saved) {
      try {
        const config = JSON.parse(saved);
        Object.entries(config).forEach(([key, value]) => {
          const element = document.getElementById(key);
          if (element) element.value = value;
        });
      } catch (e) {
        console.warn('Failed to load saved configuration:', e);
      }
    }
  }
  
  // Export for other modules
  export { ResourceOptimizerEngine, initializeResourceOptimizer };