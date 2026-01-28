import requests
import json
import time
import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class Location:
    latitude: float
    longitude: float
    address: str = ""

@dataclass
class Hospital:
    name: str
    location: Location
    phone: str = ""
    distance_km: float = 0.0
    rating: float = 0.0

@dataclass
class EmergencyRoute:
    origin: Location
    destination: Hospital
    distance_km: float
    duration_minutes: int
    instructions: List[str]
    route_coordinates: List[Tuple[float, float]]
    created_at: datetime

class AzureMapsEmergencySystem:
    def __init__(self, subscription_key: str):
        """
        Initialize Azure Maps Emergency System
        
        Args:
            subscription_key: Your Azure Maps subscription key
        """
        self.subscription_key = subscription_key
        self.base_url = "https://atlas.microsoft.com"
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.search_url = f"{self.base_url}/search/poi/json"
        self.route_url = f"{self.base_url}/route/directions/json"
        self.geocoding_url = f"{self.base_url}/search/address/json"
        
        # Emergency settings
        self.max_search_radius_km = 50  # Maximum search radius for hospitals
        self.max_hospitals_to_find = 5  # Maximum number of hospitals to retrieve
        
        print("🏥 Azure Maps Emergency System initialized")
        print(f"🔑 Subscription key: {subscription_key[:8]}...")
    
    def get_current_location_mock(self) -> Location:
        """
        Mock GPS location - Replace with actual GPS integration
        Returns a sample location (Microsoft Campus, Redmond)
        """
        # In real implementation, you would use:
        # - gpsd library for GPS
        # - Mobile device GPS API
        # - Vehicle's built-in GPS system
        
        return Location(
            latitude=47.6062,  # Redmond, WA (Microsoft Campus)
            longitude=-122.2021,
            address="Microsoft Campus, Redmond, WA"
        )
    
    def get_current_location_by_ip(self) -> Optional[Location]:
        """
        Get approximate location using IP geolocation as fallback
        """
        try:
            # Using a free IP geolocation service
            response = requests.get("http://ip-api.com/json/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    return Location(
                        latitude=data['lat'],
                        longitude=data['lon'],
                        address=f"{data['city']}, {data['regionName']}, {data['country']}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to get IP location: {e}")
        
        return None
    
    def find_nearest_hospitals(self, current_location: Location) -> List[Hospital]:
        """
        Find nearest hospitals using Azure Maps Search API
        
        Args:
            current_location: Current GPS location
            
        Returns:
            List of nearby hospitals sorted by distance
        """
        hospitals = []
        
        try:
            # Prepare search parameters
            params = {
                'api-version': '1.0',
                'subscription-key': self.subscription_key,
                'query': 'hospital',
                'lat': current_location.latitude,
                'lon': current_location.longitude,
                'radius': self.max_search_radius_km * 1000,  # Convert to meters
                'limit': self.max_hospitals_to_find,
                'categorySet': '7321'  # Hospital category in Azure Maps
            }
            
            # Make API request
            response = requests.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse results
            for result in data.get('results', []):
                try:
                    hospital_location = Location(
                        latitude=result['position']['lat'],
                        longitude=result['position']['lon'],
                        address=result.get('address', {}).get('freeformAddress', 'Unknown Address')
                    )
                    
                    # Calculate distance
                    distance_km = self._calculate_distance(
                        current_location.latitude, current_location.longitude,
                        hospital_location.latitude, hospital_location.longitude
                    )
                    
                    hospital = Hospital(
                        name=result.get('poi', {}).get('name', 'Unknown Hospital'),
                        location=hospital_location,
                        phone=result.get('poi', {}).get('phone', 'N/A'),
                        distance_km=distance_km,
                        rating=result.get('score', 0.0)
                    )
                    
                    hospitals.append(hospital)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing hospital result: {e}")
                    continue
            
            # Sort by distance
            hospitals.sort(key=lambda h: h.distance_km)
            
            self.logger.info(f"Found {len(hospitals)} hospitals within {self.max_search_radius_km}km")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Azure Maps Search API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in hospital search: {e}")
        
        return hospitals
    
    def generate_emergency_route(self, origin: Location, destination: Hospital) -> Optional[EmergencyRoute]:
        """
        Generate emergency route using Azure Maps Route API
        
        Args:
            origin: Starting location
            destination: Target hospital
            
        Returns:
            EmergencyRoute object with route details
        """
        try:
            # Prepare route parameters
            params = {
                'api-version': '1.0',
                'subscription-key': self.subscription_key,
                'query': f"{origin.latitude},{origin.longitude}:{destination.location.latitude},{destination.location.longitude}",
                'travelMode': 'car',
                'routeType': 'fastest',
                'traffic': 'true',
                'instructionsType': 'text'
            }
            
            # Make API request
            response = requests.get(self.route_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse route data
            route_data = data['routes'][0]
            summary = route_data['summary']
            
            # Extract route coordinates
            route_coordinates = []
            for leg in route_data['legs']:
                for point in leg['points']:
                    route_coordinates.append((point['latitude'], point['longitude']))
            
            # Extract turn-by-turn instructions
            instructions = []
            for leg in route_data['legs']:
                for instruction in leg.get('instructions', []):
                    instructions.append(instruction.get('message', ''))
            
            # Create emergency route object
            emergency_route = EmergencyRoute(
                origin=origin,
                destination=destination,
                distance_km=summary['lengthInMeters'] / 1000,
                duration_minutes=summary['travelTimeInSeconds'] // 60,
                instructions=instructions,
                route_coordinates=route_coordinates,
                created_at=datetime.now()
            )
            
            self.logger.info(f"Route generated: {emergency_route.distance_km:.1f}km, {emergency_route.duration_minutes}min")
            
            return emergency_route
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Azure Maps Route API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in route generation: {e}")
        
        return None
    
    def handle_emergency_situation(self, driver_state: str, confidence: float) -> Optional[EmergencyRoute]:
        """
        Main emergency handler - finds hospital and generates route
        
        Args:
            driver_state: Current driver consciousness state
            confidence: Confidence level of the detection
            
        Returns:
            EmergencyRoute if successful, None otherwise
        """
        print(f"\n🚨 EMERGENCY SITUATION DETECTED 🚨")
        print(f"Driver State: {driver_state}")
        print(f"Confidence: {confidence:.2f}")
        print("🔍 Locating nearest hospital...")
        
        # Step 1: Get current location
        current_location = self.get_current_location_mock()
        if not current_location:
            # Fallback to IP-based location
            current_location = self.get_current_location_by_ip()
            if not current_location:
                self.logger.error("Failed to determine current location")
                return None
        
        print(f"📍 Current location: {current_location.address}")
        
        # Step 2: Find nearest hospitals
        hospitals = self.find_nearest_hospitals(current_location)
        if not hospitals:
            self.logger.error("No hospitals found in the area")
            return None
        
        # Step 3: Select the nearest hospital
        nearest_hospital = hospitals[0]
        print(f"🏥 Nearest hospital: {nearest_hospital.name}")
        print(f"📍 Hospital location: {nearest_hospital.location.address}")
        print(f"📏 Distance: {nearest_hospital.distance_km:.1f}km")
        
        # Step 4: Generate emergency route
        print("🛣️  Generating emergency route...")
        emergency_route = self.generate_emergency_route(current_location, nearest_hospital)
        
        if emergency_route:
            print(f"✅ Emergency route generated successfully!")
            print(f"⏱️  Estimated travel time: {emergency_route.duration_minutes} minutes")
            print(f"📏 Total distance: {emergency_route.distance_km:.1f}km")
            
            # Save route to file
            self.save_emergency_route(emergency_route)
            
            # Display route instructions
            self.display_route_instructions(emergency_route)
            
            return emergency_route
        else:
            print("❌ Failed to generate emergency route")
            return None
    
    def save_emergency_route(self, route: EmergencyRoute):
        """Save emergency route to JSON file"""
        try:
            route_data = {
                'timestamp': route.created_at.isoformat(),
                'origin': {
                    'latitude': route.origin.latitude,
                    'longitude': route.origin.longitude,
                    'address': route.origin.address
                },
                'destination': {
                    'name': route.destination.name,
                    'latitude': route.destination.location.latitude,
                    'longitude': route.destination.location.longitude,
                    'address': route.destination.location.address,
                    'phone': route.destination.phone
                },
                'route_info': {
                    'distance_km': route.distance_km,
                    'duration_minutes': route.duration_minutes,
                    'instructions': route.instructions
                },
                'route_coordinates': route.route_coordinates
            }
            
            filename = f"emergency_route_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(route_data, f, indent=2)
            
            print(f"💾 Emergency route saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save emergency route: {e}")
    
    def display_route_instructions(self, route: EmergencyRoute, max_instructions: int = 5):
        """Display first few route instructions"""
        print(f"\n🧭 Route Instructions to {route.destination.name}:")
        print("-" * 50)
        
        for i, instruction in enumerate(route.instructions[:max_instructions]):
            if instruction.strip():
                print(f"{i+1}. {instruction}")
        
        if len(route.instructions) > max_instructions:
            print(f"... and {len(route.instructions) - max_instructions} more instructions")
        
        print("-" * 50)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        
        Returns:
            Distance in kilometers
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r

# Example usage and testing
def test_azure_maps_emergency():
    """Test function for Azure Maps Emergency System"""
    # Replace with your actual Azure Maps subscription key
    AZURE_SUBSCRIPTION_KEY = "YOUR_AZURE_MAPS_SUBSCRIPTION_KEY_HERE"
    
    # Initialize the emergency system
    emergency_system = AzureMapsEmergencySystem(AZURE_SUBSCRIPTION_KEY)
    
    # Simulate emergency situation
    emergency_route = emergency_system.handle_emergency_situation(
        driver_state="UNCONSCIOUS",
        confidence=0.95
    )
    
    if emergency_route:
        print(f"\n✅ Emergency response completed successfully!")
    else:
        print(f"\n❌ Emergency response failed!")

if __name__ == "__main__":
    test_azure_maps_emergency()