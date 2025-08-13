"""
Sublimation & Outgassing Engine Service
Calculates gas emission from the object surface due to sublimation
"""
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from scipy.constants import R, N_A
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests

# Add shared module to path
sys.path.append('/app/shared')

from shared.models import OutgassingData, Vector3D, MaterialProperties, TemperatureMap
from shared.database import get_db_manager, init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sublimation & Outgassing Engine", version="1.0.0")

# Initialize database
init_database()


class OutgassingRequest(BaseModel):
    """Request model for outgassing simulation"""
    simulation_id: str
    material_type: str
    temperature_data: Dict[str, Any]


class OutgassingResponse(BaseModel):
    """Response model for outgassing data"""
    simulation_id: str
    outgassing_data: List[Dict[str, Any]]
    message: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "outgassing-engine", "timestamp": datetime.now()}


@app.post("/calculate-outgassing", response_model=OutgassingResponse)
async def calculate_outgassing(request: OutgassingRequest, background_tasks: BackgroundTasks):
    """Calculate sublimation and outgassing rates"""
    try:
        # Start background task for outgassing calculation
        background_tasks.add_task(
            calculate_sublimation_rates,
            request.simulation_id,
            request.material_type,
            request.temperature_data
        )
        
        return OutgassingResponse(
            simulation_id=request.simulation_id,
            outgassing_data=[],
            message="Outgassing calculation started in background"
        )
        
    except Exception as e:
        logger.error(f"Error in outgassing calculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outgassing-data/{simulation_id}")
async def get_outgassing_data(simulation_id: str):
    """Get outgassing data for a specific simulation"""
    try:
        # This would retrieve stored outgassing data
        # For now, return a placeholder
        return {
            "simulation_id": simulation_id,
            "outgassing_data": [],
            "message": "Outgassing data retrieval not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving outgassing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_sublimation_rates(simulation_id: str, material_type: str, temperature_data: Dict[str, Any]):
    """Calculate sublimation and outgassing rates"""
    try:
        logger.info(f"Starting outgassing calculation for {simulation_id}")
        
        # Get material properties
        material_props = get_material_properties(material_type)
        
        # Parse temperature data
        temperatures = np.array(temperature_data["temperatures"])
        surface_points = np.array(temperature_data["surface_points"])
        timestamps = temperature_data["timestamps"]
        
        # Calculate outgassing for each time step
        outgassing_results = []
        
        for i, timestamp in enumerate(timestamps):
            temp_map = TemperatureMap(
                timestamp=datetime.fromisoformat(timestamp),
                temperatures=temperatures[i],
                surface_points=surface_points,
                max_temperature=np.max(temperatures[i]),
                min_temperature=np.min(temperatures[i]),
                average_temperature=np.mean(temperatures[i])
            )
            
            # Calculate outgassing at this time
            outgassing = calculate_outgassing_at_time(temp_map, material_props)
            outgassing_results.append(outgassing)
            
            logger.info(f"Calculated outgassing for {timestamp}")
        
        logger.info(f"Completed outgassing calculation for {simulation_id}")
        
    except Exception as e:
        logger.error(f"Error in sublimation calculation: {e}")


def get_material_properties(material_type: str) -> MaterialProperties:
    """Get material properties from database"""
    db_manager = get_db_manager()
    query = """
    SELECT * FROM materials WHERE material_type = %s
    """
    result = db_manager.execute_query(query, (material_type,))
    
    if not result:
        raise ValueError(f"Material type {material_type} not found")
    
    material_data = result[0]
    
    return MaterialProperties(
        material_type=material_data['material_type'],
        density=material_data['density'],
        heat_capacity=material_data['heat_capacity'],
        sublimation_temperature=material_data['sublimation_temperature'],
        thermal_conductivity=material_data['thermal_conductivity'],
        tensile_strength=material_data['tensile_strength'],
        albedo=material_data['albedo'],
        emissivity=material_data['emissivity']
    )


def calculate_outgassing_at_time(temp_map: TemperatureMap, material_props: MaterialProperties) -> OutgassingData:
    """Calculate outgassing at a specific time"""
    
    # Calculate gas flux at each surface point
    gas_flux = []
    velocity_vectors = []
    
    for i, temp in enumerate(temp_map.temperatures):
        # Calculate sublimation rate using Clausius-Clapeyron equation
        flux = calculate_sublimation_flux(temp, material_props)
        gas_flux.append(flux)
        
        # Calculate gas velocity (simplified)
        velocity = calculate_gas_velocity(temp, material_props)
        velocity_vectors.append(velocity)
    
    gas_flux = np.array(gas_flux)
    velocity_vectors = np.array(velocity_vectors)
    
    # Calculate total mass loss rate
    # Assuming uniform surface area per point
    surface_area_per_point = material_props.density * 1e-6  # m² per point (simplified)
    total_mass_loss = np.sum(gas_flux) * surface_area_per_point
    
    return OutgassingData(
        timestamp=temp_map.timestamp,
        gas_flux=gas_flux,
        velocity_vectors=velocity_vectors,
        total_mass_loss=total_mass_loss,
        surface_points=temp_map.surface_points,
        material_type=material_props.material_type
    )


def calculate_sublimation_flux(temperature: float, material_props: MaterialProperties) -> float:
    """Calculate sublimation flux using Clausius-Clapeyron equation"""
    
    # Sublimation temperature (K)
    T_sub = material_props.sublimation_temperature
    
    # If temperature is below sublimation temperature, no sublimation
    if temperature < T_sub:
        return 0.0
    
    # Molar mass (kg/mol) - simplified for different materials
    if material_props.material_type == "solid_nitrogen":
        molar_mass = 0.028  # N2
        heat_of_sublimation = 5.56e3  # J/mol (N2)
    elif material_props.material_type == "solid_hydrogen":
        molar_mass = 0.002  # H2
        heat_of_sublimation = 0.9e3  # J/mol (H2)
    elif material_props.material_type == "ice":
        molar_mass = 0.018  # H2O
        heat_of_sublimation = 51.0e3  # J/mol (H2O)
    else:
        # Default values for other materials
        molar_mass = 0.050  # kg/mol
        heat_of_sublimation = 20.0e3  # J/mol
    
    # Universal gas constant
    R_gas = 8.314  # J/(mol·K)
    
    # Vapor pressure using Clausius-Clapeyron equation
    # P = P₀ * exp(-ΔH_sub / (R * T))
    # where P₀ is a reference pressure
    
    # Reference pressure at sublimation temperature
    P_0 = 1.0e5  # Pa (1 atm)
    
    # Vapor pressure
    vapor_pressure = P_0 * np.exp(-heat_of_sublimation / (R_gas * temperature))
    
    # Sublimation flux (kg/(m²·s))
    # Using kinetic theory of gases
    flux = vapor_pressure * np.sqrt(molar_mass / (2 * np.pi * R_gas * temperature))
    
    return flux


def calculate_gas_velocity(temperature: float, material_props: MaterialProperties) -> np.ndarray:
    """Calculate gas velocity vector (simplified)"""
    
    # If no sublimation, no gas velocity
    if temperature < material_props.sublimation_temperature:
        return np.array([0.0, 0.0, 0.0])
    
    # Thermal velocity (simplified)
    # v = sqrt(3kT/m)
    k_boltzmann = 1.381e-23  # J/K
    
    # Molar mass in kg
    if material_props.material_type == "solid_nitrogen":
        molar_mass = 0.028
    elif material_props.material_type == "solid_hydrogen":
        molar_mass = 0.002
    elif material_props.material_type == "ice":
        molar_mass = 0.018
    else:
        molar_mass = 0.050
    
    # Molecular mass
    molecular_mass = molar_mass / N_A
    
    # Thermal velocity magnitude
    velocity_magnitude = np.sqrt(3 * k_boltzmann * temperature / molecular_mass)
    
    # Random direction (simplified - in reality would depend on surface normal)
    # For now, assume gas escapes radially outward
    velocity = np.array([velocity_magnitude, 0.0, 0.0])
    
    return velocity


def calculate_thrust_from_outgassing(outgassing_data: OutgassingData) -> float:
    """Calculate thrust force from outgassing"""
    
    # Thrust = mass flow rate * exhaust velocity
    # F = ṁ * v
    
    # Average exhaust velocity
    avg_velocity = np.mean([np.linalg.norm(v) for v in outgassing_data.velocity_vectors])
    
    # Mass flow rate (kg/s)
    mass_flow_rate = outgassing_data.total_mass_loss
    
    # Thrust (N)
    thrust = mass_flow_rate * avg_velocity
    
    return thrust


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
