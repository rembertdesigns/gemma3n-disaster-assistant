import uuid
from datetime import datetime
from app.sentiment_utils import analyze_sentiment  # Make sure this exists

# Simulated peer store
active_broadcasts = {}

def start_broadcast(message: str, location: dict, severity: str = "High"):
    """
    Start a broadcast message and enrich it with sentiment metadata.
    """
    broadcast_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # 🔍 Analyze emotional sentiment
    sentiment_result = analyze_sentiment(message)

    active_broadcasts[broadcast_id] = {
        "id": broadcast_id,
        "message": message,
        "location": location,
        "severity": severity,
        "timestamp": timestamp,
        "sentiment": sentiment_result.get("sentiment", "Unknown"),
        "tone": sentiment_result.get("tone", "Unknown"),
        "escalation": sentiment_result.get("escalation", "Unknown")
    }
    return active_broadcasts[broadcast_id]

def discover_nearby_broadcasts(current_location: dict, radius_km: float = 5.0):
    """
    Simulate discovering nearby broadcasts (mock filter by location radius).
    """
    return list(active_broadcasts.values())

def stop_broadcast(broadcast_id: str):
    """
    Remove a broadcast by ID.
    """
    if broadcast_id in active_broadcasts:
        del active_broadcasts[broadcast_id]
        return {"status": "stopped", "id": broadcast_id}
    return {"error": "broadcast not found"}