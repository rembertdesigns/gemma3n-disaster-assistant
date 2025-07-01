import { sendP2PBroadcast } from "./p2p/fallback-webrtc.js";

async function fetchRiskPrediction() {
  let lat = 34.05;
  let lon = -118.25;

  if ("geolocation" in navigator) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        lat = position.coords.latitude;
        lon = position.coords.longitude;
        console.log("📍 User location acquired:", lat, lon);
        sendPredictionRequest(lat, lon);
      },
      (error) => {
        console.warn("⚠️ Geolocation error:", error.message);
        sendPredictionRequest(lat, lon); // fallback
      }
    );
  } else {
    sendPredictionRequest(lat, lon);
  }
}

async function sendPredictionRequest(lat, lon) {
  const weather = {
    temperature: 42,
    wind_speed: 60,
    rainfall: 20
  };
  const hazardType = "wildfire";

  try {
    const response = await fetch("/predict-risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location: { lat, lon },
        weather,
        hazard_type: hazardType
      })
    });

    const data = await response.json();
    data.location = { lat, lon }; // attach location if missing

    updateRiskUI(data);
    analyzeSentiment(data.severity); // 🧠 Analyze tone

  } catch (error) {
    console.error("❌ Error fetching risk prediction:", error);
    document.getElementById("risk-score").textContent = "Risk Score: Error";
    document.getElementById("severity").textContent = "Severity: Error";
    document.getElementById("resources").textContent = "Resources: Failed to load.";
  }
}

function updateRiskUI(data) {
  document.getElementById("risk-score").innerHTML = `Risk Score: <strong>${data.risk_score}</strong>`;
  document.getElementById("severity").innerHTML = `Severity: <strong>${data.severity}</strong>`;
  document.getElementById("resources").textContent = JSON.stringify(data.suggested_resources, null, 2);

  if (data.risk_score >= 0.8) {
    triggerBroadcast(data);

    const broadcastPayload = {
      message: `⚠️ High disaster risk (${data.severity}) at location!`,
      severity: data.severity,
      location: data.location
    };
    sendP2PBroadcast(broadcastPayload);
  }
}

async function triggerBroadcast(data) {
  const payload = {
    message: `🚨 Emergency Risk Detected (${data.severity})`,
    location: data.location,
    severity: data.severity
  };

  try {
    const response = await fetch("/broadcast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    console.log("📢 Broadcast triggered:", result);

    const alertBanner = document.getElementById("broadcast-alert");
    if (alertBanner) {
      alertBanner.style.display = "block";
      alertBanner.textContent = `🚨 Broadcast triggered: ${data.severity} risk near your location.`;
    }
  } catch (error) {
    console.error("❌ Failed to trigger broadcast:", error);
  }
}

// 🧠 Gemma-powered Sentiment Analysis
async function analyzeSentiment(text) {
  try {
    const response = await fetch("/analyze-sentiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const result = await response.json();
    document.getElementById("sentiment-result").innerHTML = `Sentiment: <strong>${result.sentiment}</strong>`;
    document.getElementById("escalation-level").innerHTML = `Escalation Level: <strong>${result.escalation}</strong>`;
  } catch (err) {
    console.warn("⚠️ Sentiment analysis failed:", err);
    document.getElementById("sentiment-result").textContent = "Sentiment: Unknown";
    document.getElementById("escalation-level").textContent = "Escalation Level: N/A";
  }
}

document.addEventListener("DOMContentLoaded", fetchRiskPrediction); 