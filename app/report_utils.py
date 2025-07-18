"""
📄 Enhanced Report Utils with Map Integration

Integrates map generation and enhanced location data into PDF reports.
Builds on the existing report_utils.py functionality.
"""

from fastapi.responses import FileResponse
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime, timezone
import os
import uuid
import json
import base64
import logging

# Import our new map utilities
from .map_utils import (
    map_utils, 
    generate_static_map, 
    get_coordinate_formats, 
    get_emergency_resources,
    get_map_metadata,
    MapConfig
)

logger = logging.getLogger(__name__)

def generate_report_pdf(data: dict, output_path=None):
    """
    Generate comprehensive PDF report with enhanced timestamp, GPS data, and MAP INTEGRATION
    
    NEW FEATURES:
    - Static map images embedded in PDFs
    - Multiple coordinate system formats
    - Emergency resource proximity data
    - Enhanced location analysis
    - Map metadata and accuracy indicators
    """
    env = Environment(loader=FileSystemLoader("app/templates"))
    template = env.get_template("report_template.html")

    # Generate unique filename if not provided
    if output_path is None:
        report_id = str(uuid.uuid4())[:8]
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/emergency_report_{timestamp_str}_{report_id}.pdf"

    # Enhanced timestamp processing (existing)
    timestamp_data = process_timestamps(data)
    
    # ENHANCED GPS processing with MAP INTEGRATION
    gps_data = process_gps_data_with_maps(data)
    
    # Extract and process other data
    hazards_list = process_hazards(data.get("hazards", []))
    checklist_data = process_checklist(data.get("checklist", []))
    
    # Resolve image path
    image_data = process_image_data(data.get("image_url"))
    
    # Enhanced metadata for PDF
    pdf_metadata = {
        "title": f"Emergency Report - {timestamp_data['incident_date']}",
        "subject": f"Incident at {gps_data['display_location']}",
        "creator": "Disaster Response Assistant",
        "producer": "EdgeAI Emergency System",
        "keywords": f"emergency, {', '.join(hazards_list)}, severity-{data.get('severity', 'unknown')}",
        "creation_date": timestamp_data['utc_iso'],
        "modification_date": timestamp_data['utc_iso'],
        # Custom emergency metadata
        "emergency_id": data.get("id", str(uuid.uuid4())),
        "incident_severity": str(data.get("severity", "unknown")),
        "gps_coordinates": gps_data['coordinates_string'],
        "gps_accuracy": gps_data['accuracy_string'],
        "response_priority": determine_priority(data.get("severity", 1)),
        "ai_analysis_included": str(bool(data.get("ai_analysis"))),
        # NEW: Map metadata
        "map_included": str(gps_data.get('map_data_available', False)),
        "emergency_resources_count": str(len(gps_data.get('emergency_resources', []))),
        "coordinate_formats": str(len(gps_data.get('coord_formats', {}))),
    }

    # Render HTML with enhanced data including MAP DATA
    html_content = template.render(
        # Timestamp data
        **timestamp_data,
        
        # ENHANCED GPS and location data with MAPS
        **gps_data,
        
        # Core incident data
        location=data.get("location", "Unknown Location"),
        hazards=", ".join(hazards_list),
        severity=data.get("severity", "N/A"),
        notes=data.get("notes", ""),
        
        # Enhanced data
        checklist=checklist_data,
        image_data=image_data,
        pdf_metadata=pdf_metadata,
        
        # AI Analysis data if available
        ai_analysis=process_ai_analysis(data.get("ai_analysis")),
        
        # Report generation metadata
        report_id=data.get("id", str(uuid.uuid4())),
        generated_by="EdgeAI Disaster Response System",
        report_version="2.1",  # Updated version with map integration
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate PDF with metadata
    html_doc = HTML(string=html_content, base_url=".")
    
    # Write PDF with enhanced metadata
    html_doc.write_pdf(
        output_path,
        pdf_version="1.7",  # Modern PDF version
        pdf_metadata=pdf_metadata,
        optimize_images=True,
        compress=True
    )
    
    # Log report generation with map data
    log_report_generation(output_path, data, timestamp_data, gps_data)
    
    return output_path

def process_gps_data_with_maps(data):
    """
    ENHANCED GPS processing with MAP INTEGRATION
    
    NEW FEATURES:
    - Static map generation for PDF embedding
    - Multiple coordinate system formats using map_utils
    - Emergency resource proximity calculation
    - Enhanced location metadata
    """
    
    # Extract coordinates (existing logic)
    coordinates = data.get("coordinates", [])
    if isinstance(coordinates, str):
        try:
            lat, lng = map(float, coordinates.split(","))
            coordinates = [lat, lng]
        except:
            coordinates = []
    
    latitude = coordinates[0] if len(coordinates) >= 1 else None
    longitude = coordinates[1] if len(coordinates) >= 2 else None
    
    # GPS metadata from client (existing)
    gps_accuracy = data.get("gps_accuracy")
    gps_source = data.get("gps_source", "Unknown")
    
    # NEW: Generate coordinate strings using enhanced map utilities
    if latitude and longitude:
        coordinates_string = f"{latitude:.6f}, {longitude:.6f}"
        coordinates_display = f"Lat: {latitude:.5f}, Lng: {longitude:.5f}"
        
        # NEW: Use map_utils for enhanced coordinate formats
        try:
            coord_formats = get_coordinate_formats(latitude, longitude)
            logger.info(f"Generated coordinate formats for {latitude}, {longitude}")
        except Exception as e:
            logger.warning(f"Failed to generate coordinate formats: {e}")
            coord_formats = {
                'decimal_degrees': coordinates_display,
                'dms': convert_to_dms(latitude, longitude),  # Fallback to existing
            }
        
        # NEW: Generate static map for PDF
        map_image_data = None
        map_data_available = False
        try:
            # Configure map for PDF
            map_config = MapConfig(
                width=600,
                height=400,
                zoom=15,
                map_type="roadmap",
                marker_color="red"
            )
            
            map_image_bytes = generate_static_map(latitude, longitude, map_config)
            if map_image_bytes:
                # Convert to base64 for HTML embedding
                map_image_base64 = base64.b64encode(map_image_bytes).decode('utf-8')
                map_image_data = {
                    'data_url': f"data:image/png;base64,{map_image_base64}",
                    'width': map_config.width,
                    'height': map_config.height,
                    'zoom': map_config.zoom,
                    'service': map_utils.preferred_service.value
                }
                map_data_available = True
                logger.info(f"Successfully generated static map for {latitude}, {longitude}")
            else:
                logger.warning(f"Failed to generate static map for {latitude}, {longitude}")
        except Exception as e:
            logger.error(f"Map generation error: {e}")
            map_image_data = None
        
        # NEW: Find emergency resources
        emergency_resources = []
        try:
            emergency_resources = get_emergency_resources(latitude, longitude, radius_km=25)
            logger.info(f"Found {len(emergency_resources)} emergency resources")
        except Exception as e:
            logger.warning(f"Failed to find emergency resources: {e}")
        
        # NEW: Get comprehensive map metadata
        location_metadata = {}
        try:
            location_metadata = get_map_metadata(latitude, longitude)
            logger.info("Generated location metadata")
        except Exception as e:
            logger.warning(f"Failed to generate location metadata: {e}")
        
        # GPS accuracy assessment (existing + enhanced)
        if gps_accuracy:
            accuracy_string = f"±{gps_accuracy}m"
            accuracy_level = get_accuracy_level(gps_accuracy)
        else:
            accuracy_string = "Unknown accuracy"
            accuracy_level = "unknown"
            
        # Enhanced location display
        display_location = data.get("location", coordinates_display)
        
    else:
        # No coordinates available
        coordinates_string = "Not available"
        coordinates_display = "Location not specified"
        accuracy_string = "N/A"
        accuracy_level = "none"
        display_location = data.get("location", "Unknown Location")
        coord_formats = {}
        map_image_data = None
        map_data_available = False
        emergency_resources = []
        location_metadata = {}

    # Return enhanced GPS data structure
    return {
        # Basic coordinate data (existing)
        "coordinates": coordinates,
        "latitude": latitude,
        "longitude": longitude,
        "coordinates_string": coordinates_string,
        "coordinates_display": coordinates_display,
        "display_location": display_location,
        "gps_accuracy": gps_accuracy,
        "accuracy_string": accuracy_string,
        "accuracy_level": accuracy_level,
        "gps_source": gps_source,
        
        # ENHANCED: Multiple coordinate formats
        "coord_formats": coord_formats,
        
        # NEW: Map integration
        "map_image": map_image_data,
        "map_data_available": map_data_available,
        
        # NEW: Emergency resources
        "emergency_resources": emergency_resources,
        "nearest_hospital": emergency_resources[0] if emergency_resources and emergency_resources[0].type == 'hospital' else None,
        "nearest_fire_station": next((r for r in emergency_resources if r.type == 'fire_station'), None),
        "nearest_police": next((r for r in emergency_resources if r.type == 'police'), None),
        
        # NEW: Location analysis
        "location_metadata": location_metadata,
        
        # Enhanced emergency grid reference
        "emergency_grid_ref": generate_emergency_grid_ref(latitude, longitude) if latitude and longitude else "N/A",
    }

def generate_map_preview_data(latitude: float, longitude: float):
    """
    Generate map preview data for web interface
    Used by main.py routes for live map updates
    """
    try:
        map_metadata = get_map_metadata(latitude, longitude)
        coordinate_formats = get_coordinate_formats(latitude, longitude)
        emergency_resources = get_emergency_resources(latitude, longitude)
        
        return {
            'success': True,
            'coordinates': {
                'latitude': latitude,
                'longitude': longitude,
                'formats': coordinate_formats
            },
            'emergency_resources': [
                {
                    'name': resource.name,
                    'type': resource.type,
                    'distance_km': round(resource.distance_km, 2),
                    'estimated_time': resource.estimated_time,
                    'coordinates': [resource.latitude, resource.longitude]
                }
                for resource in emergency_resources[:10]  # Limit to first 10
            ],
            'metadata': map_metadata
        }
    except Exception as e:
        logger.error(f"Failed to generate map preview data: {e}")
        return {
            'success': False,
            'error': str(e),
            'coordinates': {'latitude': latitude, 'longitude': longitude}
        }

def generate_static_map_endpoint(latitude: float, longitude: float, 
                               width: int = 600, height: int = 400, zoom: int = 15):
    """
    Generate static map for API endpoint
    Used by main.py /api/static-map route
    """
    try:
        config = MapConfig(
            width=width,
            height=height,
            zoom=zoom,
            map_type="roadmap"
        )
        
        map_bytes = generate_static_map(latitude, longitude, config)
        
        if map_bytes:
            return {
                'success': True,
                'image_data': base64.b64encode(map_bytes).decode('utf-8'),
                'content_type': 'image/png',
                'size': len(map_bytes),
                'config': {
                    'width': width,
                    'height': height,
                    'zoom': zoom,
                    'service': map_utils.preferred_service.value
                }
            }
        else:
            return {
                'success': False,
                'error': 'Failed to generate map image',
                'fallback_available': True
            }
    except Exception as e:
        logger.error(f"Static map endpoint error: {e}")
        return {
            'success': False,
            'error': str(e),
            'fallback_available': True
        }

# Keep all existing functions from the original report_utils.py
def process_timestamps(data):
    """Process and enhance timestamp information"""
    
    # Get incident timestamp (prefer provided, fallback to current)
    if data.get("timestamp"):
        try:
            incident_dt = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        except:
            incident_dt = datetime.now(timezone.utc)
    else:
        incident_dt = datetime.now(timezone.utc)
    
    # Generate comprehensive timestamp data
    utc_now = datetime.now(timezone.utc)
    
    return {
        # Primary timestamps
        "incident_timestamp": incident_dt.isoformat(),
        "utc_iso": incident_dt.isoformat(),
        "utc_timestamp": incident_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "local_timestamp": incident_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        
        # Human-readable formats
        "incident_date": incident_dt.strftime("%Y-%m-%d"),
        "incident_time": incident_dt.strftime("%H:%M:%S"),
        "incident_datetime": incident_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "formatted_date": incident_dt.strftime("%B %d, %Y"),
        "formatted_time": incident_dt.strftime("%I:%M %p"),
        
        # Report generation timestamps
        "generated_at": utc_now.isoformat(),
        "generated_utc": utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "generated_local": utc_now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        
        # Time calculations
        "time_since_incident": calculate_time_since(incident_dt, utc_now),
        "day_of_week": incident_dt.strftime("%A"),
        "month_name": incident_dt.strftime("%B"),
        "is_night_time": incident_dt.hour < 6 or incident_dt.hour > 18,
        "is_weekend": incident_dt.weekday() >= 5,
        
        # Emergency response timing
        "response_urgency": calculate_response_urgency(incident_dt, utc_now),
        "golden_hour_remaining": calculate_golden_hour(incident_dt, utc_now),
    }

def process_hazards(hazards_data):
    """Process hazards list into standardized format"""
    if isinstance(hazards_data, str):
        return [h.strip() for h in hazards_data.split(",") if h.strip()]
    elif isinstance(hazards_data, list):
        return [str(h).strip() for h in hazards_data if str(h).strip()]
    return []

def process_checklist(checklist_data):
    """Process checklist data with enhanced formatting"""
    if not checklist_data:
        return []
    
    processed_checklist = []
    for item in checklist_data:
        if isinstance(item, dict):
            processed_checklist.append({
                "item": item.get("item", ""),
                "completed": item.get("completed", False),
                "assigned_to": item.get("assigned_to", "Unassigned"),
                "priority": item.get("priority", "normal"),
                "estimated_time": item.get("estimated_time", ""),
                "completion_timestamp": item.get("completion_timestamp", ""),
            })
        else:
            processed_checklist.append({
                "item": str(item),
                "completed": False,
                "assigned_to": "Unassigned",
                "priority": "normal",
                "estimated_time": "",
                "completion_timestamp": "",
            })
    
    return processed_checklist

def process_image_data(image_url):
    """Process image data with metadata"""
    if not image_url:
        return None
    
    # Handle different image URL formats
    if image_url.startswith("uploaded://"):
        filename = image_url.replace("uploaded://", "")
        image_path = os.path.join("uploads", filename)
        if os.path.exists(image_path):
            return {
                "url": f"/{image_path}",
                "type": "uploaded",
                "path": image_path,
                "exists": True,
                "size": get_file_size(image_path),
                "modified": get_file_modified_time(image_path),
            }
    elif image_url.startswith("offline_blob://"):
        return {
            "url": None,
            "type": "offline_blob",
            "note": "Image stored offline, will be available after sync",
            "exists": False,
        }
    elif image_url.startswith("http"):
        return {
            "url": image_url,
            "type": "external",
            "exists": True,
        }
    else:
        # Assume local file
        if os.path.exists(image_url):
            return {
                "url": f"/{image_url}",
                "type": "local",
                "path": image_url,
                "exists": True,
                "size": get_file_size(image_url),
            }
    
    return None

def process_ai_analysis(ai_analysis_data):
    """Process AI analysis data for PDF inclusion"""
    if not ai_analysis_data:
        return None
    
    return {
        "severity_score": ai_analysis_data.get("severity", {}).get("overall", "N/A"),
        "confidence": ai_analysis_data.get("severity", {}).get("confidence", 0),
        "text_analysis": ai_analysis_data.get("textAnalysis", {}),
        "image_analysis": ai_analysis_data.get("imageAnalysis", {}),
        "recommendations": ai_analysis_data.get("recommendations", []),
        "processing_time": ai_analysis_data.get("processingTime", 0),
        "analysis_timestamp": ai_analysis_data.get("timestamp", ""),
    }

# Helper functions for coordinate conversions (FALLBACK - map_utils preferred)
def convert_to_dms(latitude, longitude):
    """Convert decimal degrees to degrees, minutes, seconds format"""
    def decimal_to_dms(decimal_deg):
        degrees = int(abs(decimal_deg))
        minutes_float = (abs(decimal_deg) - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        return degrees, minutes, seconds
    
    lat_deg, lat_min, lat_sec = decimal_to_dms(latitude)
    lng_deg, lng_min, lng_sec = decimal_to_dms(longitude)
    
    lat_dir = "N" if latitude >= 0 else "S"
    lng_dir = "E" if longitude >= 0 else "W"
    
    return f"{lat_deg}°{lat_min}'{lat_sec:.2f}\"{lat_dir}, {lng_deg}°{lng_min}'{lng_sec:.2f}\"{lng_dir}"

def generate_emergency_grid_ref(latitude, longitude):
    """Generate emergency services grid reference"""
    # Simplified emergency grid reference for first responders
    try:
        grid_lat = chr(65 + int((latitude + 90) / 10))  # A-R
        grid_lng = chr(65 + int((longitude + 180) / 20))  # A-R
        sub_lat = int(((latitude + 90) % 10) * 10)
        sub_lng = int(((longitude + 180) % 20) * 10)
        return f"EMRG-{grid_lat}{grid_lng}-{sub_lat}{sub_lng}"
    except:
        return "Grid ref unavailable"

def get_accuracy_level(accuracy_meters):
    """Determine GPS accuracy level"""
    if accuracy_meters <= 3:
        return "excellent"
    elif accuracy_meters <= 10:
        return "good"
    elif accuracy_meters <= 50:
        return "moderate"
    else:
        return "poor"

def calculate_time_since(incident_time, current_time):
    """Calculate human-readable time since incident"""
    delta = current_time - incident_time
    
    if delta.total_seconds() < 60:
        return f"{int(delta.total_seconds())} seconds ago"
    elif delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() // 60)} minutes ago"
    elif delta.total_seconds() < 86400:
        return f"{int(delta.total_seconds() // 3600)} hours ago"
    else:
        return f"{int(delta.days)} days ago"

def calculate_response_urgency(incident_time, current_time):
    """Calculate response urgency based on time elapsed"""
    delta = current_time - incident_time
    minutes_elapsed = delta.total_seconds() / 60
    
    if minutes_elapsed < 5:
        return "immediate"
    elif minutes_elapsed < 15:
        return "urgent"
    elif minutes_elapsed < 60:
        return "high"
    elif minutes_elapsed < 240:  # 4 hours
        return "moderate"
    else:
        return "routine"

def calculate_golden_hour(incident_time, current_time):
    """Calculate remaining time in the 'golden hour' for medical emergencies"""
    delta = current_time - incident_time
    minutes_elapsed = delta.total_seconds() / 60
    golden_hour_minutes = 60
    
    remaining = golden_hour_minutes - minutes_elapsed
    
    if remaining > 0:
        return f"{int(remaining)} minutes remaining"
    else:
        return f"Exceeded by {int(abs(remaining))} minutes"

def determine_priority(severity):
    """Determine response priority based on severity"""
    try:
        severity_num = float(severity)
        if severity_num >= 8:
            return "CRITICAL"
        elif severity_num >= 6:
            return "HIGH"
        elif severity_num >= 4:
            return "MEDIUM"
        else:
            return "LOW"
    except:
        return "UNKNOWN"

def get_file_size(file_path):
    """Get human-readable file size"""
    try:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        else:
            return f"{size_bytes/(1024**2):.1f} MB"
    except:
        return "Unknown size"

def get_file_modified_time(file_path):
    """Get file modification time"""
    try:
        mtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mtime, timezone.utc).isoformat()
    except:
        return "Unknown"

def log_report_generation(output_path, data, timestamp_data, gps_data):
    """Log report generation for audit trail with MAP DATA"""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_file": output_path,
        "incident_id": data.get("id"),
        "incident_timestamp": timestamp_data["incident_timestamp"],
        "location": gps_data["display_location"],
        "coordinates": gps_data["coordinates_string"],
        "severity": data.get("severity"),
        "generated_by": "EdgeAI System v2.1",
        "file_size": get_file_size(output_path) if os.path.exists(output_path) else "0 B",
        # NEW: Map integration logging
        "map_included": gps_data.get("map_data_available", False),
        "emergency_resources_found": len(gps_data.get("emergency_resources", [])),
        "coordinate_formats_generated": len(gps_data.get("coord_formats", {})),
    }
    
    # Write to log file
    log_file = "outputs/report_generation.log"
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Could not write to log file: {e}")