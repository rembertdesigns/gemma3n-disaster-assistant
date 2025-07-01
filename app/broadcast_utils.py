import time
import uuid
from datetime import datetime

# Simulated peer store
active_broadcasts = {}

def start_broadcast(message: str, location: dict, severity: str = "High"):
    """
    Simulate starting a broadcast message.
    """
    broadcast_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    active_broadcasts[broadcast_id] = {
        "id": broadcast_id,
        "message": message,
        "location": location,
        "severity": severity,
        "timestamp": timestamp
    }
    return active_broadcasts[broadcast_id]

def discover_nearby_broadcasts(current_location: dict, radius_km: float = 5.0):
    """
    Simulate discovering nearby broadcasts (mock filter by location radius).
    """
    # TODO: Replace with real geo filtering
    return list(active_broadcasts.values())

def stop_broadcast(broadcast_id: str):
    """
    Remove a broadcast by ID.
    """
    if broadcast_id in active_broadcasts:
        del active_broadcasts[broadcast_id]
        return {"status": "stopped", "id": broadcast_id}
    return {"error": "broadcast not found"}