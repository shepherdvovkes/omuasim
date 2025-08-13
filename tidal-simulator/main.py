"""
Tidal Force Simulator Service
Calculates tidal forces and stresses on the object surface
"""
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from scipy.spatial.distance import cdist
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests

# Add shared module to path
sys.path.append('/app/shared')

from shared.models import TidalStressMap, Vector3D, ObjectGeometry, OrbitalState
from shared.database import get_db_manager, init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tidal Force Simulator", version="1.0.0")

# Initialize database
init_database()


class TidalSimulationRequest(BaseModel):
    """Request model for tidal force simulation"""
    simulation_id: str
    object_geometry: Dict[str, Any]
    start_time: str
    end_time: str


class TidalResponse(BaseModel):
    """Response model for tidal force data"""
    simulation_id: str
    stress_maps: List[Dict[str, Any]]
    message: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tidal-simulator", "timestamp": datetime.now()}


@app.post("/calculate-tidal-forces", response_model=TidalResponse)
async def calculate_tidal_forces(request: TidalSimulationRequest, background_tasks: BackgroundTasks):
    """Calculate tidal forces and stresses for the object"""
    try:
        # Parse geometry
        geometry = ObjectGeometry(
            shape=request.object_geometry["shape"],
            dimensions=Vector3D(**request.object_geometry["dimensions"]),
            aspect_ratio=request.object_geometry["aspect_ratio"],
            surface_area=request.object_geometry["surface_area"],
            volume=request.object_geometry["volume"]
        )
        
        # Start background task for tidal force calculation
        background_tasks.add_task(
            calculate_tidal_stresses,
            request.simulation_id,
            geometry,
            request.start_time,
            request.end_time
        )
        
        return TidalResponse(
            simulation_id=request.simulation_id,
            stress_maps=[],
            message="Tidal force calculation started in background"
        )
        
    except Exception as e:
        logger.error(f"Error in tidal force calculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tidal-data/{simulation_id}")
async def get_tidal_data(simulation_id: str):
    """Get tidal force data for a specific simulation"""
    try:
        # This would retrieve stored tidal stress data
        # For now, return a placeholder
        return {
            "simulation_id": simulation_id,
            "stress_maps": [],
            "message": "Tidal data retrieval not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving tidal data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_tidal_stresses(simulation_id: str, geometry: ObjectGeometry, 
                           start_time: str, end_time: str):
    """Calculate tidal stresses across the object surface"""
    try:
        logger.info(f"Starting tidal force calculation for {simulation_id}")
        
        # Get orbital data from database
        db_manager = get_db_manager()
        query = """
        SELECT * FROM orbital_states 
        WHERE simulation_id = %s 
        ORDER BY timestamp
        """
        orbital_data = db_manager.execute_query(query, (simulation_id,))
        
        # Generate surface mesh for the object
        surface_points = generate_surface_mesh(geometry)
        
        # Calculate tidal stresses for each orbital state
        for orbital_record in orbital_data:
            # Convert database record to OrbitalState
            orbital_state = OrbitalState(
                timestamp=orbital_record['timestamp'],
                position=Vector3D(
                    orbital_record['position_x'],
                    orbital_record['position_y'],
                    orbital_record['position_z']
                ),
                velocity=Vector3D(
                    orbital_record['velocity_x'],
                    orbital_record['velocity_y'],
                    orbital_record['velocity_z']
                ),
                acceleration=Vector3D(
                    orbital_record['acceleration_x'],
                    orbital_record['acceleration_y'],
                    orbital_record['acceleration_z']
                )
            )
            
            # Calculate tidal stress map
            stress_map = calculate_stress_at_time(orbital_state, geometry, surface_points)
            
            # Store stress map (would implement database storage)
            logger.info(f"Calculated tidal stresses for {orbital_state.timestamp}")
        
        logger.info(f"Completed tidal force calculation for {simulation_id}")
        
    except Exception as e:
        logger.error(f"Error in tidal stress calculation: {e}")


def generate_surface_mesh(geometry: ObjectGeometry) -> np.ndarray:
    """Generate surface mesh points for the object"""
    if geometry.shape == "cigar":
        return generate_cigar_surface(geometry)
    elif geometry.shape == "sphere":
        return generate_sphere_surface(geometry)
    else:
        return generate_ellipsoid_surface(geometry)


def generate_cigar_surface(geometry: ObjectGeometry) -> np.ndarray:
    """Generate surface points for a cigar-shaped object"""
    # Simplified cigar shape: cylinder with hemispherical ends
    length = geometry.dimensions.x
    radius = geometry.dimensions.y / 2
    
    # Generate points along the length
    n_length = 20
    n_circumference = 16
    
    points = []
    
    # Cylindrical part
    for i in range(n_length):
        z = (i / (n_length - 1) - 0.5) * length
        
        for j in range(n_circumference):
            angle = 2 * np.pi * j / n_circumference
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y, z])
    
    # Hemispherical ends
    for end in [-1, 1]:  # -1 for left end, 1 for right end
        z_center = end * length / 2
        
        for i in range(n_circumference):
            for j in range(n_circumference // 2):
                phi = np.pi * j / (n_circumference // 2)
                theta = 2 * np.pi * i / n_circumference
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = z_center + end * radius * np.cos(phi)
                
                points.append([x, y, z])
    
    return np.array(points)


def generate_sphere_surface(geometry: ObjectGeometry) -> np.ndarray:
    """Generate surface points for a spherical object"""
    radius = geometry.dimensions.x / 2
    n_points = 100
    
    points = []
    for i in range(n_points):
        for j in range(n_points):
            phi = np.pi * i / n_points
            theta = 2 * np.pi * j / n_points
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            points.append([x, y, z])
    
    return np.array(points)


def generate_ellipsoid_surface(geometry: ObjectGeometry) -> np.ndarray:
    """Generate surface points for an ellipsoidal object"""
    a = geometry.dimensions.x / 2
    b = geometry.dimensions.y / 2
    c = geometry.dimensions.z / 2
    
    n_points = 50
    
    points = []
    for i in range(n_points):
        for j in range(n_points):
            phi = np.pi * i / n_points
            theta = 2 * np.pi * j / n_points
            
            x = a * np.sin(phi) * np.cos(theta)
            y = b * np.sin(phi) * np.sin(theta)
            z = c * np.cos(phi)
            
            points.append([x, y, z])
    
    return np.array(points)


def calculate_stress_at_time(orbital_state: OrbitalState, geometry: ObjectGeometry, 
                           surface_points: np.ndarray) -> TidalStressMap:
    """Calculate tidal stress at a specific time"""
    
    # Distance from Sun (in AU)
    distance_au = np.sqrt(
        orbital_state.position.x**2 + 
        orbital_state.position.y**2 + 
        orbital_state.position.z**2
    )
    
    # Convert to meters
    distance_m = distance_au * 1.496e11  # 1 AU in meters
    
    # Solar mass in kg
    solar_mass = 1.989e30
    
    # Gravitational constant
    G = 6.67430e-11
    
    # Tidal force gradient (d²Φ/dr²)
    # For a point mass, this is 2GM/r³
    tidal_gradient = 2 * G * solar_mass / (distance_m ** 3)
    
    # Calculate stress tensor at each surface point
    stress_tensors = []
    principal_stresses = []
    
    for point in surface_points:
        # Distance from object center to surface point
        r_point = np.linalg.norm(point)
        
        # Tidal stress tensor (simplified)
        # In a real calculation, this would be more complex
        stress_tensor = np.zeros((3, 3))
        
        # Principal stresses (simplified model)
        # σ_r = tidal_gradient * r_point
        sigma_r = tidal_gradient * r_point
        
        # For a cigar shape, stress varies with orientation
        if geometry.shape == "cigar":
            # Higher stress along the long axis
            sigma_axial = sigma_r * 2.0
            sigma_radial = sigma_r * 0.5
        else:
            sigma_axial = sigma_r
            sigma_radial = sigma_r
        
        principal_stress = np.array([sigma_axial, sigma_radial, sigma_radial])
        principal_stresses.append(principal_stress)
        
        # Build stress tensor (simplified)
        stress_tensor[0, 0] = sigma_axial
        stress_tensor[1, 1] = sigma_radial
        stress_tensor[2, 2] = sigma_radial
        stress_tensors.append(stress_tensor)
    
    stress_tensors = np.array(stress_tensors)
    principal_stresses = np.array(principal_stresses)
    
    # Calculate max/min stresses
    max_tensile = np.max(principal_stresses)
    max_compressive = np.min(principal_stresses)
    
    return TidalStressMap(
        timestamp=orbital_state.timestamp,
        stress_tensor=stress_tensors,
        principal_stresses=principal_stresses,
        surface_points=surface_points,
        max_tensile_stress=max_tensile,
        max_compressive_stress=max_compressive
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
