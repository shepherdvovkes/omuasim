"""
Shared data models for the 'Oumuamua simulator
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum


class MaterialType(Enum):
    """Types of materials that can be simulated"""
    SOLID_NITROGEN = "solid_nitrogen"
    SOLID_HYDROGEN = "solid_hydrogen"
    ROCK = "rock"
    ICE = "ice"
    CARBONACEOUS = "carbonaceous"


@dataclass
class Vector3D:
    """3D vector representation"""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector3D':
        return cls(arr[0], arr[1], arr[2])


@dataclass
class OrbitalState:
    """Orbital state vector at a specific time"""
    timestamp: datetime
    position: Vector3D  # AU
    velocity: Vector3D  # AU/day
    acceleration: Vector3D  # AU/day²


@dataclass
class MaterialProperties:
    """Physical properties of a material"""
    material_type: MaterialType
    density: float  # kg/m³
    heat_capacity: float  # J/(kg·K)
    sublimation_temperature: float  # K
    thermal_conductivity: float  # W/(m·K)
    tensile_strength: float  # Pa
    albedo: float  # Reflectivity (0-1)
    emissivity: float  # Emissivity (0-1)


@dataclass
class ObjectGeometry:
    """Geometric properties of the object"""
    shape: str  # "cigar", "sphere", "ellipsoid"
    dimensions: Vector3D  # Length, width, height in meters
    aspect_ratio: float  # Length to width ratio
    surface_area: float  # m²
    volume: float  # m³


@dataclass
class TidalStressMap:
    """Map of tidal stresses across the object surface"""
    timestamp: datetime
    stress_tensor: np.ndarray  # 3x3 stress tensor at each surface point
    principal_stresses: np.ndarray  # Principal stress values
    surface_points: np.ndarray  # Surface point coordinates
    max_tensile_stress: float  # Maximum tensile stress
    max_compressive_stress: float  # Maximum compressive stress


@dataclass
class TemperatureMap:
    """Temperature distribution across the object"""
    timestamp: datetime
    temperatures: np.ndarray  # Temperature at each surface point
    surface_points: np.ndarray  # Surface point coordinates
    max_temperature: float  # Maximum temperature
    min_temperature: float  # Minimum temperature
    average_temperature: float  # Average temperature


@dataclass
class OutgassingData:
    """Data about gas emission from the surface"""
    timestamp: datetime
    gas_flux: np.ndarray  # Gas flux at each surface point (kg/m²/s)
    velocity_vectors: np.ndarray  # Gas velocity vectors
    total_mass_loss: float  # Total mass loss rate (kg/s)
    surface_points: np.ndarray  # Surface point coordinates
    material_type: MaterialType


@dataclass
class ThrustVector:
    """Thrust vector resulting from outgassing"""
    timestamp: datetime
    force: Vector3D  # Thrust force vector (N)
    magnitude: float  # Thrust magnitude (N)
    direction: Vector3D  # Thrust direction (unit vector)
    origin_point: Vector3D  # Point of application


@dataclass
class SimulationParameters:
    """Parameters for a simulation run"""
    material_type: MaterialType
    object_geometry: ObjectGeometry
    start_time: datetime
    end_time: datetime
    time_step: float  # Time step in days
    solar_distance_range: Tuple[float, float]  # Min/max distance from Sun (AU)


@dataclass
class SimulationResult:
    """Results of a complete simulation"""
    parameters: SimulationParameters
    base_trajectory: List[OrbitalState]
    corrected_trajectory: List[OrbitalState]
    tidal_stresses: List[TidalStressMap]
    temperatures: List[TemperatureMap]
    outgassing: List[OutgassingData]
    thrust_vectors: List[ThrustVector]
    deviation_from_observed: float  # Deviation from observed trajectory
    confidence_score: float  # Confidence in the result (0-1)


@dataclass
class ObservationData:
    """Real observational data for comparison"""
    timestamp: datetime
    position: Vector3D
    velocity: Vector3D
    uncertainty_position: Vector3D
    uncertainty_velocity: Vector3D
    observatory: str
    instrument: str
