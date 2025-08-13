-- Initialize materials database with physical properties
-- Data for 'Oumuamua simulation materials

-- Clear existing data
DELETE FROM materials;

-- Insert material properties
INSERT INTO materials (material_type, density, heat_capacity, sublimation_temperature, thermal_conductivity, tensile_strength, albedo, emissivity) VALUES
-- Solid Nitrogen (N2) - proposed by Jackson & Desch (2021)
('solid_nitrogen', 1026.0, 1040.0, 63.15, 0.025, 1.0e6, 0.64, 0.95),

-- Solid Hydrogen (H2) - alternative hypothesis
('solid_hydrogen', 86.0, 14300.0, 13.8, 0.18, 1.0e5, 0.5, 0.9),

-- Rock (typical asteroid material)
('rock', 3000.0, 800.0, 1500.0, 2.0, 5.0e7, 0.15, 0.9),

-- Ice (water ice)
('ice', 917.0, 2100.0, 273.15, 2.22, 1.0e6, 0.6, 0.9),

-- Carbonaceous material (cometary)
('carbonaceous', 1500.0, 1200.0, 800.0, 0.5, 2.0e6, 0.05, 0.9);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_materials_type ON materials(material_type);
CREATE INDEX IF NOT EXISTS idx_orbital_states_timestamp ON orbital_states(timestamp);
CREATE INDEX IF NOT EXISTS idx_orbital_states_simulation_id ON orbital_states(simulation_id);
CREATE INDEX IF NOT EXISTS idx_simulation_results_id ON simulation_results(simulation_id);
CREATE INDEX IF NOT EXISTS idx_observation_data_timestamp ON observation_data(timestamp);
