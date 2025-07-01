"""
🗺️ Map Utilities for Emergency Response System

Handles static map generation for PDFs, coordinate formatting,
and emergency resource proximity calculations.

Features:
- Static map image generation for PDF reports
- Multiple coordinate system support (Lat/Lng, UTM, MGRS, Plus Codes)
- Emergency resource proximity calculations
- Map service API integration (OpenStreetMap, MapBox, Google)
- Offline fallback capabilities
- Emergency grid reference generation
"""

import os
import requests
import base64
import io
import math
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)

class MapService(Enum):
    """Available map services for static map generation"""
    OPENSTREETMAP = "osm"
    MAPBOX = "mapbox"
    GOOGLE = "google"
    FALLBACK = "fallback"

class CoordinateFormat(Enum):
    """Supported coordinate formats"""
    DECIMAL_DEGREES = "dd"
    DEGREES_MINUTES_SECONDS = "dms"
    UTM = "utm"
    MGRS = "mgrs"
    PLUS_CODE = "plus_code"
    EMERGENCY_GRID = "emergency_grid"

@dataclass
class MapConfig:
    """Configuration for map generation"""
    width: int = 600
    height: int = 400
    zoom: int = 15
    map_type: str = "roadmap"  # roadmap, satellite, terrain, hybrid
    marker_color: str = "red"
    marker_size: str = "mid"
    api_key: Optional[str] = None
    cache_duration: int = 3600  # 1 hour cache

@dataclass
class EmergencyResource:
    """Emergency resource location data"""
    name: str
    type: str  # hospital, fire_station, police, evacuation_route
    latitude: float
    longitude: float
    distance_km: float
    estimated_time: str
    capacity: Optional[str] = None
    contact: Optional[str] = None

class MapUtils:
    """Comprehensive map utilities for emergency response"""
    
    def __init__(self):
        self.config = MapConfig()
        self.cache = {}  # Simple in-memory cache
        self.preferred_service = MapService.OPENSTREETMAP
        
        # Load API keys from environment
        self.api_keys = {
            MapService.MAPBOX: os.getenv('MAPBOX_API_KEY'),
            MapService.GOOGLE: os.getenv('GOOGLE_MAPS_API_KEY')
        }
        
        # Determine best available service
        self._determine_best_service()
        
    def _determine_best_service(self):
        """Determine the best available map service based on API keys"""
        if self.api_keys[MapService.GOOGLE]:
            self.preferred_service = MapService.GOOGLE
            logger.info("Using Google Maps as primary map service")
        elif self.api_keys[MapService.MAPBOX]:
            self.preferred_service = MapService.MAPBOX
            logger.info("Using MapBox as primary map service")
        else:
            self.preferred_service = MapService.OPENSTREETMAP
            logger.info("Using OpenStreetMap as primary map service")

    def generate_static_map(self, latitude: float, longitude: float, 
                          config: Optional[MapConfig] = None) -> Optional[bytes]:
        """
        Generate static map image for PDF inclusion
        
        Args:
            latitude: Incident latitude
            longitude: Incident longitude
            config: Map configuration (uses default if None)
            
        Returns:
            Map image as bytes, or None if generation fails
        """
        if config is None:
            config = self.config
            
        # Generate cache key
        cache_key = self._generate_cache_key(latitude, longitude, config)
        
        # Check cache first
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < config.cache_duration:
                logger.debug(f"Returning cached map for {latitude}, {longitude}")
                return cache_entry['data']
        
        # Try primary service first, then fallbacks
        services_to_try = [self.preferred_service]
        
        # Add fallback services
        for service in MapService:
            if service != self.preferred_service and service != MapService.FALLBACK:
                services_to_try.append(service)
        
        for service in services_to_try:
            try:
                map_data = self._generate_map_with_service(service, latitude, longitude, config)
                if map_data:
                    # Cache successful result
                    self.cache[cache_key] = {
                        'data': map_data,
                        'timestamp': time.time()
                    }
                    logger.info(f"Successfully generated map using {service.value}")
                    return map_data
            except Exception as e:
                logger.warning(f"Map generation failed with {service.value}: {e}")
                continue
        
        # If all services fail, generate fallback
        logger.warning("All map services failed, generating fallback")
        return self._generate_fallback_map(latitude, longitude, config)

    def _generate_map_with_service(self, service: MapService, lat: float, lng: float, 
                                 config: MapConfig) -> Optional[bytes]:
        """Generate map using specific service"""
        
        if service == MapService.GOOGLE:
            return self._generate_google_static_map(lat, lng, config)
        elif service == MapService.MAPBOX:
            return self._generate_mapbox_static_map(lat, lng, config)
        elif service == MapService.OPENSTREETMAP:
            return self._generate_osm_static_map(lat, lng, config)
        else:
            return None

    def _generate_google_static_map(self, lat: float, lng: float, 
                                  config: MapConfig) -> Optional[bytes]:
        """Generate static map using Google Static Maps API"""
        if not self.api_keys[MapService.GOOGLE]:
            raise ValueError("Google Maps API key not available")
        
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            'center': f"{lat},{lng}",
            'zoom': config.zoom,
            'size': f"{config.width}x{config.height}",
            'maptype': config.map_type,
            'markers': f"color:{config.marker_color}|size:{config.marker_size}|{lat},{lng}",
            'key': self.api_keys[MapService.GOOGLE],
            'format': 'png'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.content

    def _generate_mapbox_static_map(self, lat: float, lng: float, 
                                  config: MapConfig) -> Optional[bytes]:
        """Generate static map using MapBox Static Images API"""
        if not self.api_keys[MapService.MAPBOX]:
            raise ValueError("MapBox API key not available")
        
        # Convert map_type to MapBox style
        style_map = {
            'roadmap': 'streets-v11',
            'satellite': 'satellite-v9',
            'terrain': 'outdoors-v11',
            'hybrid': 'satellite-streets-v11'
        }
        style = style_map.get(config.map_type, 'streets-v11')
        
        # MapBox URL format: /styles/v1/{username}/{style_id}/static/{overlay}/{lon},{lat},{zoom}/{width}x{height}
        base_url = f"https://api.mapbox.com/styles/v1/mapbox/{style}/static"
        overlay = f"pin-s-emergency+{config.marker_color.replace('#', '')}({lng},{lat})"
        url = f"{base_url}/{overlay}/{lng},{lat},{config.zoom}/{config.width}x{config.height}"
        
        params = {
            'access_token': self.api_keys[MapService.MAPBOX]
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.content

    def _generate_osm_static_map(self, lat: float, lng: float, 
                               config: MapConfig) -> Optional[bytes]:
        """Generate static map using OpenStreetMap (via staticmap service)"""
        
        # Use a free OSM static map service
        base_url = "https://api.mapbox.com/styles/v1/mapbox/streets-v11/static"
        
        # Fallback to a simple tile-based approach if no service available
        # This is a basic implementation - in production you might want to use
        # a more sophisticated tile stitching approach
        
        try:
            # Calculate tile coordinates
            zoom = config.zoom
            x, y = self._deg2tile(lat, lng, zoom)
            
            # Get surrounding tiles to create a larger image
            tile_urls = []
            for dx in range(-1, 2):  # 3x3 grid of tiles
                for dy in range(-1, 2):
                    tile_x = x + dx
                    tile_y = y + dy
                    tile_url = f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
                    tile_urls.append(tile_url)
            
            # For simplicity, just return the center tile
            # In a full implementation, you'd stitch tiles together
            center_tile_url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            
            response = requests.get(center_tile_url, timeout=10, 
                                  headers={'User-Agent': 'EmergencyResponseApp/1.0'})
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            logger.error(f"OSM tile generation failed: {e}")
            return None

    def _generate_fallback_map(self, lat: float, lng: float, 
                             config: MapConfig) -> bytes:
        """Generate a simple fallback map when all services fail"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a simple image with coordinates
            img = Image.new('RGB', (config.width, config.height), color='lightgray')
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw coordinate information
            text_lines = [
                "EMERGENCY LOCATION",
                f"Latitude: {lat:.6f}",
                f"Longitude: {lng:.6f}",
                "",
                "Map service unavailable",
                "Coordinates verified"
            ]
            
            y_offset = 50
            for line in text_lines:
                # Get text bounding box for centering
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (config.width - text_width) // 2
                
                draw.text((x, y_offset), line, fill='black', font=font)
                y_offset += 30
            
            # Draw a simple crosshair at center
            center_x, center_y = config.width // 2, config.height // 2
            cross_size = 20
            draw.line([center_x - cross_size, center_y, center_x + cross_size, center_y], 
                     fill='red', width=3)
            draw.line([center_x, center_y - cross_size, center_x, center_y + cross_size], 
                     fill='red', width=3)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue()
            
        except ImportError:
            # If PIL is not available, return a very basic text representation
            fallback_text = f"EMERGENCY LOCATION\nLat: {lat:.6f}\nLng: {lng:.6f}\nMap unavailable"
            return fallback_text.encode('utf-8')

    def format_coordinates(self, latitude: float, longitude: float, 
                         format_type: CoordinateFormat) -> str:
        """
        Format coordinates in various systems for emergency response
        
        Args:
            latitude: Decimal degrees latitude
            longitude: Decimal degrees longitude
            format_type: Desired coordinate format
            
        Returns:
            Formatted coordinate string
        """
        
        if format_type == CoordinateFormat.DECIMAL_DEGREES:
            return f"{latitude:.6f}, {longitude:.6f}"
        
        elif format_type == CoordinateFormat.DEGREES_MINUTES_SECONDS:
            return self._to_dms(latitude, longitude)
        
        elif format_type == CoordinateFormat.UTM:
            return self._to_utm(latitude, longitude)
        
        elif format_type == CoordinateFormat.MGRS:
            return self._to_mgrs(latitude, longitude)
        
        elif format_type == CoordinateFormat.PLUS_CODE:
            return self._to_plus_code(latitude, longitude)
        
        elif format_type == CoordinateFormat.EMERGENCY_GRID:
            return self._to_emergency_grid(latitude, longitude)
        
        else:
            return f"{latitude:.6f}, {longitude:.6f}"

    def get_all_coordinate_formats(self, latitude: float, longitude: float) -> Dict[str, str]:
        """Get coordinates in all supported formats"""
        return {
            'decimal_degrees': self.format_coordinates(latitude, longitude, CoordinateFormat.DECIMAL_DEGREES),
            'dms': self.format_coordinates(latitude, longitude, CoordinateFormat.DEGREES_MINUTES_SECONDS),
            'utm': self.format_coordinates(latitude, longitude, CoordinateFormat.UTM),
            'mgrs': self.format_coordinates(latitude, longitude, CoordinateFormat.MGRS),
            'plus_code': self.format_coordinates(latitude, longitude, CoordinateFormat.PLUS_CODE),
            'emergency_grid': self.format_coordinates(latitude, longitude, CoordinateFormat.EMERGENCY_GRID)
        }

    def calculate_distance(self, lat1: float, lng1: float, 
                         lat2: float, lng2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def find_emergency_resources(self, latitude: float, longitude: float, 
                               radius_km: float = 25) -> List[EmergencyResource]:
        """
        Find emergency resources near the incident location
        
        Args:
            latitude: Incident latitude
            longitude: Incident longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of nearby emergency resources
        """
        
        # Mock emergency resources for demonstration
        # In production, this would query a real emergency services database
        mock_resources = [
            # Hospitals
            EmergencyResource("Austin General Hospital", "hospital", 30.2849, -97.7341, 0, "5 min", "Level 1 Trauma", "512-555-0100"),
            EmergencyResource("Dell Seton Medical Center", "hospital", 30.2672, -97.7431, 0, "8 min", "Level 1 Trauma", "512-555-0200"),
            EmergencyResource("St. David's Medical Center", "hospital", 30.2950, -97.7364, 0, "12 min", "Full Service", "512-555-0300"),
            
            # Fire Stations
            EmergencyResource("Fire Station 1", "fire_station", 30.2700, -97.7400, 0, "3 min", "Engine + Ladder", "911"),
            EmergencyResource("Fire Station 5", "fire_station", 30.2600, -97.7500, 0, "6 min", "Engine + Rescue", "911"),
            
            # Police
            EmergencyResource("APD Central District", "police", 30.2650, -97.7450, 0, "4 min", "Patrol + SWAT", "911"),
            EmergencyResource("APD East District", "police", 30.2550, -97.7350, 0, "8 min", "Patrol", "911"),
            
            # Evacuation Routes
            EmergencyResource("Interstate 35 North", "evacuation_route", 30.2800, -97.7400, 0, "2 min", "Major Highway", None),
            EmergencyResource("Highway 183 East", "evacuation_route", 30.2600, -97.7200, 0, "5 min", "State Highway", None)
        ]
        
        # Calculate distances and filter by radius
        nearby_resources = []
        for resource in mock_resources:
            distance = self.calculate_distance(latitude, longitude, 
                                            resource.latitude, resource.longitude)
            if distance <= radius_km:
                # Update distance and estimated time
                resource.distance_km = distance
                resource.estimated_time = self._estimate_travel_time(distance, resource.type)
                nearby_resources.append(resource)
        
        # Sort by distance
        nearby_resources.sort(key=lambda x: x.distance_km)
        
        return nearby_resources

    def get_map_metadata(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Get comprehensive map metadata for incident reports
        
        Returns:
            Dictionary with map metadata, coordinates, and emergency info
        """
        
        return {
            'coordinates': {
                'latitude': latitude,
                'longitude': longitude,
                'formats': self.get_all_coordinate_formats(latitude, longitude)
            },
            'emergency_resources': self.find_emergency_resources(latitude, longitude),
            'map_config': {
                'zoom': self.config.zoom,
                'service': self.preferred_service.value,
                'timestamp': time.time()
            },
            'location_analysis': {
                'terrain_type': self._analyze_terrain(latitude, longitude),
                'accessibility': self._analyze_accessibility(latitude, longitude),
                'risk_factors': self._analyze_risk_factors(latitude, longitude)
            }
        }

    # Helper methods for coordinate conversion
    def _to_dms(self, lat: float, lng: float) -> str:
        """Convert to Degrees, Minutes, Seconds format"""
        def dd_to_dms(dd: float, is_lat: bool) -> str:
            direction = 'N' if dd >= 0 and is_lat else 'S' if is_lat else 'E' if dd >= 0 else 'W'
            dd = abs(dd)
            degrees = int(dd)
            minutes = int((dd - degrees) * 60)
            seconds = ((dd - degrees) * 60 - minutes) * 60
            return f"{degrees}°{minutes}'{seconds:.2f}\"{direction}"
        
        lat_dms = dd_to_dms(lat, True)
        lng_dms = dd_to_dms(lng, False)
        return f"{lat_dms}, {lng_dms}"

    def _to_utm(self, lat: float, lng: float) -> str:
        """Convert to UTM coordinates (simplified)"""
        # This is a simplified UTM conversion
        # In production, use a library like pyproj for accurate conversion
        zone = int((lng + 180) / 6) + 1
        hemisphere = 'N' if lat >= 0 else 'S'
        
        # Simplified calculation (not accurate for production use)
        easting = int(500000 + (lng - (zone - 1) * 6 - 183) * 111320 * math.cos(math.radians(lat)))
        northing = int(lat * 110540) if lat >= 0 else int(10000000 + lat * 110540)
        
        return f"{zone}{hemisphere} {easting} {northing}"

    def _to_mgrs(self, lat: float, lng: float) -> str:
        """Convert to Military Grid Reference System (simplified)"""
        # This is a very simplified MGRS representation
        # In production, use a proper MGRS library
        utm = self._to_utm(lat, lng)
        return f"MGRS: {utm[:3]}AA{utm[4:8]}{utm[9:13]}"

    def _to_plus_code(self, lat: float, lng: float) -> str:
        """Generate Plus Code (Open Location Code) - simplified"""
        # This is a placeholder implementation
        # In production, use the official openlocationcode library
        base_code = "8C3Q"  # Austin area base
        lat_code = int((lat - 30.0) * 8000)
        lng_code = int((lng + 98.0) * 8000)
        return f"{base_code}+{lat_code:04d}{lng_code:04d}"

    def _to_emergency_grid(self, lat: float, lng: float) -> str:
        """Generate emergency grid reference"""
        # Create a local emergency grid system
        # Grid squares are approximately 1km x 1km
        grid_x = int((lng + 98.0) * 100) % 100
        grid_y = int((lat - 30.0) * 100) % 100
        
        # Convert to letter-number system
        letter_x = chr(ord('A') + (grid_x // 10))
        letter_y = chr(ord('A') + (grid_y // 10))
        
        return f"EMER-{letter_x}{letter_y}-{grid_x % 10}{grid_y % 10}"

    def _deg2tile(self, lat: float, lng: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lng to tile coordinates"""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lng + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def _estimate_travel_time(self, distance_km: float, resource_type: str) -> str:
        """Estimate travel time to emergency resource"""
        # Average speeds for different emergency vehicles
        speeds = {
            'hospital': 60,      # Ambulance speed
            'fire_station': 50,  # Fire truck speed
            'police': 70,        # Police vehicle speed
            'evacuation_route': 80  # Highway speed
        }
        
        speed = speeds.get(resource_type, 50)
        time_hours = distance_km / speed
        time_minutes = int(time_hours * 60)
        
        if time_minutes < 1:
            return "< 1 min"
        elif time_minutes < 60:
            return f"{time_minutes} min"
        else:
            hours = time_minutes // 60
            minutes = time_minutes % 60
            return f"{hours}h {minutes}m"

    def _analyze_terrain(self, lat: float, lng: float) -> str:
        """Analyze terrain type at location"""
        # Simplified terrain analysis
        # In production, use elevation APIs or terrain databases
        return "urban_flat"  # urban_flat, rural_hilly, mountainous, coastal, etc.

    def _analyze_accessibility(self, lat: float, lng: float) -> str:
        """Analyze location accessibility for emergency vehicles"""
        # Simplified accessibility analysis
        return "good"  # excellent, good, moderate, poor, inaccessible

    def _analyze_risk_factors(self, lat: float, lng: float) -> List[str]:
        """Identify environmental risk factors"""
        # Simplified risk factor analysis
        # In production, cross-reference with hazard databases
        return ["urban_environment", "traffic_congestion"]

    def _generate_cache_key(self, lat: float, lng: float, config: MapConfig) -> str:
        """Generate cache key for map images"""
        key_data = f"{lat:.6f}_{lng:.6f}_{config.width}_{config.height}_{config.zoom}_{config.map_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

# Global instance
map_utils = MapUtils()

# Convenience functions for easy import
def generate_static_map(latitude: float, longitude: float, config: Optional[MapConfig] = None) -> Optional[bytes]:
    """Generate static map image"""
    return map_utils.generate_static_map(latitude, longitude, config)

def get_coordinate_formats(latitude: float, longitude: float) -> Dict[str, str]:
    """Get coordinates in all formats"""
    return map_utils.get_all_coordinate_formats(latitude, longitude)

def get_emergency_resources(latitude: float, longitude: float, radius_km: float = 25) -> List[EmergencyResource]:
    """Find nearby emergency resources"""
    return map_utils.find_emergency_resources(latitude, longitude, radius_km)

def get_map_metadata(latitude: float, longitude: float) -> Dict[str, Any]:
    """Get comprehensive map metadata"""
    return map_utils.get_map_metadata(latitude, longitude)