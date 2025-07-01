import random
from datetime import datetime

# Mock risk factors by hazard type
HAZARD_WEIGHTS = {
    "flood": 0.9,
    "wildfire": 0.8,
    "earthquake": 0.85,
    "storm": 0.75,
    "heatwave": 0.7,
    "unknown": 0.5
}

# Mock weather severity scale
def weather_severity(weather_data: dict) -> float:
    wind = weather_data.get("wind_speed", 0)
    rain = weather_data.get("rainfall", 0)
    temp = weather_data.get("temperature", 25)

    score = 0.0
    if wind > 50:
        score += 0.4
    if rain > 100:
        score += 0.4
    if temp > 40 or temp < -5:
        score += 0.2

    return min(score, 1.0)

# Main risk score calculation
def calculate_risk_score(location: dict, weather_data: dict, hazard_type: str = "unknown") -> dict:
    """
    Returns a dict with:
    - risk_score (0.0 to 1.0)
    - suggested_resources
    - severity_label
    """
    lat = location.get("lat", 0)
    lon = location.get("lon", 0)
    hazard_weight = HAZARD_WEIGHTS.get(hazard_type.lower(), 0.5)
    weather_score = weather_severity(weather_data)
    
    # Randomized local threat input for demo purposes
    proximity_factor = random.uniform(0.3, 0.9)

    # Weighted risk formula
    risk_score = round(
        (hazard_weight * 0.5 + weather_score * 0.3 + proximity_factor * 0.2),
        2
    )

    severity_label = classify_severity(risk_score)
    suggested_resources = forecast_resources(risk_score, hazard_type)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "location": {"lat": lat, "lon": lon},
        "hazard_type": hazard_type,
        "risk_score": risk_score,
        "severity": severity_label,
        "suggested_resources": suggested_resources
    }

# Risk level mapping
def classify_severity(score: float) -> str:
    if score >= 0.8:
        return "Critical"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Low"

# Mock resource forecast
def forecast_resources(risk_score: float, hazard_type: str) -> dict:
    base = max(1, int(risk_score * 10))
    return {
        "med_kits": base * 5,
        "water_liters": base * 50,
        "rescue_teams": max(1, base // 2),
        "shelter_capacity": base * 20,
        "drones": 1 if hazard_type in ["wildfire", "earthquake"] else 0
    }