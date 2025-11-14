"""
QRR-Enhanced Solid Rocket Motor Internal Ballistics Simulator
==============================================================

Physics-rigorous implementation of internal ballistics using Quantum Relational
Relativity (QRR) mathematics to analyze combustion-pressure-flow coupling.

References:
- Sutton & Biblarz, "Rocket Propulsion Elements" (9th Ed)
- Saint-Robert's Law for propellant burn rate
- Ideal gas dynamics for chamber pressure

Author: Relational Relativity LLC (Robin Macomber, Bruce Stephenson)
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
import sys
import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# QRR library location on C drive
QRR_LIB_PATH = Path(r"C:\Users\Robin_B_Macomber\Relational_Relativity\QRR_Coding_and_Applications\QRR_Libraries\Python_QRR")

# Output directory on D drive
OUTPUT_DIR = Path(r"D:\RelationalRelativity\rocket_motor_ballistics_upwork\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"QRR Library: {QRR_LIB_PATH}")
print(f"Output Directory: {OUTPUT_DIR}")

# Add QRR library to Python path
sys.path.insert(0, str(QRR_LIB_PATH))

# Import QRR library
try:
    from enhanced_qrr_core import (
        QRRSystem, RelationalEntity, CoherenceDensityCalculator,
        QuantumPhenomena, QRRConstants, MathPrimitives
    )
    QRR_AVAILABLE = True
    print("QRR library loaded successfully")
except ImportError as e:
    print(f"Warning: QRR library not found. Running in physics-only mode.")
    print(f"Error: {e}")
    QRR_AVAILABLE = False

# ============================================================================
# PHYSICAL CONSTANTS AND PROPELLANT PROPERTIES
# ============================================================================

@dataclass
class PhysicalConstants:
    """Physical constants for rocket motor simulation"""
    R_UNIVERSAL = 8314.46  # J/(kmol·K)
    GAMMA = 1.25           # Specific heat ratio for solid propellant products
    MOLECULAR_WEIGHT = 25.0  # kg/kmol (typical for composite propellant)
    T_CHAMBER = 3200.0     # Combustion temperature (K)
    P_ATM = 101325.0       # Atmospheric pressure (Pa)
    
    @property
    def R_gas(self):
        """Specific gas constant (J/kg·K)"""
        return self.R_UNIVERSAL / self.MOLECULAR_WEIGHT

@dataclass
class PropellantProperties:
    """Solid propellant characteristics"""
    density: float = 1750.0      # kg/m³ (typical composite)
    a_burn: float = 0.005        # Burn rate coefficient (m/s at 1 MPa)
    n_burn: float = 0.35         # Burn rate exponent (pressure sensitivity)
    c_star: float = 1550.0       # Characteristic velocity (m/s)
    energy_density: float = 5.0e6  # Energy per unit mass (J/kg)
    
    def burn_rate(self, pressure_pa: float) -> float:
        """
        Saint-Robert's burn rate law: ṙ = a·P^n
        
        Args:
            pressure_pa: Chamber pressure in Pascals
            
        Returns:
            Burn rate in m/s
        """
        if pressure_pa <= 0:
            return 0.0
        return self.a_burn * (pressure_pa / 1e6) ** self.n_burn

@dataclass
class MotorGeometry:
    """Solid rocket motor geometry (cylindrical grain)"""
    outer_radius: float = 0.15     # m (15 cm outer radius)
    initial_port_radius: float = 0.05  # m (5 cm initial bore)
    length: float = 1.0            # m (1 meter long)
    throat_area: float = 0.001     # m² (10 cm² throat)
    expansion_ratio: float = 4.0   # Exit/throat area ratio
    
    def burning_surface_area(self, port_radius: float) -> float:
        """
        Calculate burning surface area for cylindrical grain.
        Includes internal bore surface and end faces.
        
        Args:
            port_radius: Current port radius (m)
            
        Returns:
            Burning surface area (m²)
        """
        if port_radius >= self.outer_radius:
            return 0.0
        
        # Internal cylindrical surface
        A_bore = 2 * np.pi * port_radius * self.length
        
        # Two end faces (annular areas)
        A_ends = 2 * np.pi * (self.outer_radius**2 - port_radius**2)
        
        return A_bore + A_ends
    
    def chamber_volume(self, port_radius: float) -> float:
        """
        Calculate chamber volume.
        
        Args:
            port_radius: Current port radius (m)
            
        Returns:
            Chamber volume (m³)
        """
        return np.pi * port_radius**2 * self.length
    
    def web_thickness(self, port_radius: float) -> float:
        """Remaining propellant thickness"""
        return self.outer_radius - port_radius
    
    def is_burned_out(self, port_radius: float) -> bool:
        """Check if grain is burned out (< 5% web remaining)"""
        return self.web_thickness(port_radius) < 0.05 * self.outer_radius

# ============================================================================
# QRR-ENHANCED MOTOR STATE
# ============================================================================

class QRRMotorState:
    """
    Rocket motor state with QRR relational analysis.
    
    This class tracks the coupled relationships between:
    - Chamber pressure (P)
    - Burn rate (ṙ) 
    - Mass generation rate (ṁ_gen)
    - Nozzle mass flow (ṁ_nozzle)
    - Grain geometry evolution
    
    QRR coherence measures how well these relationships maintain
    their expected coupling during the burn.
    """
    
    def __init__(self, constants: PhysicalConstants, 
                 propellant: PropellantProperties,
                 geometry: MotorGeometry):
        self.const = constants
        self.prop = propellant
        self.geom = geometry
        
        # Physical state variables
        self.time = 0.0
        self.port_radius = geometry.initial_port_radius
        self.chamber_pressure = 0.0  # Will ignite with initial pressure
        self.burn_rate = 0.0
        self.mass_gen_rate = 0.0
        self.nozzle_mass_flow = 0.0
        self.thrust = 0.0
        
        # QRR state
        if QRR_AVAILABLE:
            self.qrr_system = QRRSystem("RocketMotor")
            self._initialize_qrr_entities()
        else:
            self.qrr_system = None
        
        self.coherence = 1.0
        self.relational_energy = 0.0
        self.entanglement_strength = 0.0
        
        # History for analysis
        self.history = {
            'time': [],
            'pressure': [],
            'thrust': [],
            'burn_rate': [],
            'port_radius': [],
            'coherence': [],
            'relational_energy': [],
            'mass_flow': []
        }
    
    def _initialize_qrr_entities(self):
        """Initialize QRR relational entities for motor components"""
        # Create entities for coupled phenomena
        pressure_entity = RelationalEntity("chamber_pressure", weight=1.0)
        burn_entity = RelationalEntity("burn_rate", weight=1.0)
        flow_entity = RelationalEntity("mass_flow", weight=1.0)
        geometry_entity = RelationalEntity("grain_geometry", weight=1.0)
        
        # Add to QRR system
        self.qrr_system.add_entity(pressure_entity)
        self.qrr_system.add_entity(burn_entity)
        self.qrr_system.add_entity(flow_entity)
        self.qrr_system.add_entity(geometry_entity)
        
        # Create entanglements (strong coupling)
        self.qrr_system.create_entanglement("chamber_pressure", "burn_rate", 0.9)
        self.qrr_system.create_entanglement("burn_rate", "grain_geometry", 0.85)
        self.qrr_system.create_entanglement("chamber_pressure", "mass_flow", 0.95)
    
    def compute_nozzle_flow(self) -> float:
        """
        Compute choked nozzle mass flow using isentropic relations.
        
        Returns:
            Mass flow rate (kg/s)
        """
        if self.chamber_pressure <= self.const.P_ATM:
            return 0.0
        
        gamma = self.const.GAMMA
        R = self.const.R_gas
        T = self.const.T_CHAMBER
        
        # Choked flow coefficient
        flow_coeff = np.sqrt(gamma * (2/(gamma+1))**((gamma+1)/(gamma-1)))
        
        # Mass flow: ṁ = (P·A_t / sqrt(T)) · sqrt(γ/R) · Γ
        mass_flow = (self.chamber_pressure * self.geom.throat_area / 
                     np.sqrt(T)) * flow_coeff * np.sqrt(gamma / R)
        
        return mass_flow
    
    def compute_thrust(self) -> float:
        """
        Compute thrust using momentum and pressure contributions.
        F = ṁ·v_e + (P_e - P_a)·A_e
        
        Returns:
            Thrust (N)
        """
        if self.chamber_pressure <= self.const.P_ATM:
            return 0.0
        
        gamma = self.const.GAMMA
        R = self.const.R_gas
        T = self.const.T_CHAMBER
        
        # Exit pressure (simplified: assume optimal expansion)
        P_exit = self.const.P_ATM
        
        # Exit velocity from isentropic expansion
        v_exit = np.sqrt(2 * gamma / (gamma - 1) * R * T * 
                        (1 - (P_exit / self.chamber_pressure)**((gamma-1)/gamma)))
        
        # Momentum thrust
        thrust_momentum = self.nozzle_mass_flow * v_exit
        
        # Pressure thrust
        A_exit = self.geom.throat_area * self.geom.expansion_ratio
        thrust_pressure = (P_exit - self.const.P_ATM) * A_exit
        
        return thrust_momentum + thrust_pressure
    
    def compute_qrr_coherence(self) -> float:
        """
        Compute QRR coherence based on relational consistency.
        
        Measures how well pressure-burn-flow relationships maintain
        their expected coupling. High coherence = stable operation.
        
        Returns:
            Coherence value [0, 1]
        """
        if not QRR_AVAILABLE or self.burn_rate == 0:
            return 1.0
        
        # Expected burn rate from Saint-Robert's law
        expected_burn = self.prop.burn_rate(self.chamber_pressure)
        burn_coherence = 1.0 - abs(self.burn_rate - expected_burn) / max(expected_burn, 1e-9)
        
        # Mass flow balance coherence
        A_burn = self.geom.burning_surface_area(self.port_radius)
        expected_mass_gen = self.prop.density * A_burn * self.burn_rate
        
        if expected_mass_gen > 0:
            flow_balance = abs(self.mass_gen_rate - self.nozzle_mass_flow) / expected_mass_gen
            flow_coherence = np.exp(-flow_balance)  # Exponential decay with imbalance
        else:
            flow_coherence = 1.0
        
        # Geometric coherence (smooth regression)
        geom_coherence = 1.0 if self.port_radius < self.geom.outer_radius else 0.0
        
        # Overall system coherence (geometric mean)
        overall_coherence = (burn_coherence * flow_coherence * geom_coherence) ** (1/3)
        
        return np.clip(overall_coherence, 0.0, 1.0)
    
    def compute_relational_energy(self) -> float:
        """
        Compute relational energy - coupling strength between state variables.
        E_R = P·ṙ·ṁ (normalized)
        
        Returns:
            Relational energy (dimensionless)
        """
        if not QRR_AVAILABLE:
            return 0.0
        
        # Normalize each term
        P_norm = self.chamber_pressure / 1e7  # Typical max ~10 MPa
        r_norm = self.burn_rate / 0.01       # Typical max ~10 mm/s
        m_norm = self.nozzle_mass_flow / 10.0  # Typical max ~10 kg/s
        
        return P_norm * r_norm * m_norm
    
    def update_qrr_metrics(self):
        """Update QRR system metrics"""
        self.coherence = self.compute_qrr_coherence()
        self.relational_energy = self.compute_relational_energy()
        
        if QRR_AVAILABLE and self.qrr_system:
            # Update QRR system energy and bandwidth
            self.qrr_system.system_energy = self.relational_energy * 100
            self.qrr_system.relational_bandwidth = 1.0 / max(self.coherence, 0.01)
            self.qrr_system.update_system_state()
            
            # Get entanglement strength from network
            entanglement_matrix = self.qrr_system.entanglement_network.get_entanglement_matrix()
            if entanglement_matrix.size > 0:
                self.entanglement_strength = np.mean(entanglement_matrix[entanglement_matrix > 0])
    
    def record_history(self):
        """Record current state to history"""
        self.history['time'].append(self.time)
        self.history['pressure'].append(self.chamber_pressure / 1e6)  # MPa
        self.history['thrust'].append(self.thrust / 1000)  # kN
        self.history['burn_rate'].append(self.burn_rate * 1000)  # mm/s
        self.history['port_radius'].append(self.port_radius * 100)  # cm
        self.history['coherence'].append(self.coherence)
        self.history['relational_energy'].append(self.relational_energy)
        self.history['mass_flow'].append(self.nozzle_mass_flow)

# ============================================================================
# INTEGRATION ENGINE
# ============================================================================

class RocketMotorSimulator:
    """
    Time integration engine for rocket motor internal ballistics.
    Uses Runge-Kutta 4th order for accuracy.
    """
    
    def __init__(self, state: QRRMotorState, dt: float = 0.001):
        self.state = state
        self.dt = dt
        
    def compute_derivatives(self, port_radius: float, pressure: float) -> Tuple[float, float]:
        """
        Compute time derivatives: dR/dt and dP/dt
        
        Args:
            port_radius: Current port radius (m)
            pressure: Current chamber pressure (Pa)
            
        Returns:
            (dR/dt, dP/dt) derivatives
        """
        # Geometry
        A_burn = self.state.geom.burning_surface_area(port_radius)
        V_chamber = self.state.geom.chamber_volume(port_radius)
        
        if V_chamber == 0 or self.state.geom.is_burned_out(port_radius):
            return 0.0, -pressure / 0.01  # Rapid pressure decay
        
        # Burn rate from pressure
        burn_rate = self.state.prop.burn_rate(pressure)
        
        # Mass generation rate
        mass_gen = self.state.prop.density * A_burn * burn_rate
        
        # Nozzle mass flow (update state for flow calculation)
        self.state.chamber_pressure = pressure
        mass_flow = self.state.compute_nozzle_flow()
        
        # Net mass accumulation
        net_mass_rate = mass_gen - mass_flow
        
        # Volume change rate from regression
        dV_dt = 2 * np.pi * port_radius * self.state.geom.length * burn_rate
        
        # Pressure rate of change (ideal gas law)
        # dP/dt = (RT/V)·dm/dt - (P/V)·dV/dt
        R = self.state.const.R_gas
        T = self.state.const.T_CHAMBER
        
        dP_dt = (R * T / V_chamber) * net_mass_rate - (pressure / V_chamber) * dV_dt
        
        # Port radius rate of change
        dR_dt = burn_rate
        
        return dR_dt, dP_dt
    
    def rk4_step(self) -> bool:
        """
        Perform one RK4 integration step.
        
        Returns:
            True if simulation should continue, False if burned out
        """
        R = self.state.port_radius
        P = self.state.chamber_pressure
        dt = self.dt
        
        # Check burnout
        if self.state.geom.is_burned_out(R):
            return False
        
        # RK4 integration
        k1_R, k1_P = self.compute_derivatives(R, P)
        k2_R, k2_P = self.compute_derivatives(R + 0.5*dt*k1_R, P + 0.5*dt*k1_P)
        k3_R, k3_P = self.compute_derivatives(R + 0.5*dt*k2_R, P + 0.5*dt*k2_P)
        k4_R, k4_P = self.compute_derivatives(R + dt*k3_R, P + dt*k3_P)
        
        # Update state
        self.state.port_radius = R + (dt/6) * (k1_R + 2*k2_R + 2*k3_R + k4_R)
        self.state.chamber_pressure = max(0, P + (dt/6) * (k1_P + 2*k2_P + 2*k3_P + k4_P))
        
        # Update derived quantities
        self.state.burn_rate = self.state.prop.burn_rate(self.state.chamber_pressure)
        A_burn = self.state.geom.burning_surface_area(self.state.port_radius)
        self.state.mass_gen_rate = self.state.prop.density * A_burn * self.state.burn_rate
        self.state.nozzle_mass_flow = self.state.compute_nozzle_flow()
        self.state.thrust = self.state.compute_thrust()
        
        # Update QRR metrics
        self.state.update_qrr_metrics()
        
        # Update time
        self.state.time += dt
        
        return True
    
    def run_simulation(self, max_time: float = 10.0, record_interval: float = 0.01):
        """
        Run complete motor burn simulation.
        
        Args:
            max_time: Maximum simulation time (s)
            record_interval: Data recording interval (s)
        """
        # Ignition: Set initial pressure
        self.state.chamber_pressure = 2.0e6  # 2 MPa ignition pressure
        
        next_record_time = 0.0
        
        print(f"Starting simulation...")
        print(f"Initial conditions: P = {self.state.chamber_pressure/1e6:.2f} MPa, "
              f"R = {self.state.port_radius*100:.2f} cm")
        
        while self.state.time < max_time:
            # Integration step
            continue_sim = self.rk4_step()
            
            # Record data
            if self.state.time >= next_record_time:
                self.state.record_history()
                next_record_time += record_interval
            
            # Check termination
            if not continue_sim:
                print(f"Burnout at t = {self.state.time:.3f} s")
                break
                
            # Progress indicator
            if int(self.state.time * 10) % 10 == 0 and self.state.time > 0:
                print(f"  t = {self.state.time:.1f} s: P = {self.state.chamber_pressure/1e6:.2f} MPa, "
                      f"F = {self.state.thrust/1000:.2f} kN, C = {self.state.coherence:.3f}")
        
        print(f"Simulation complete: {len(self.state.history['time'])} data points recorded")

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def plot_results(state: QRRMotorState, save_path: str = None):
    """Generate comprehensive analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QRR-Enhanced Solid Rocket Motor Internal Ballistics', 
                 fontsize=16, fontweight='bold')
    
    time = np.array(state.history['time'])
    
    # Pressure history
    ax = axes[0, 0]
    ax.plot(time, state.history['pressure'], 'r-', linewidth=2, label='Chamber Pressure')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_title('Chamber Pressure vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Thrust history
    ax = axes[0, 1]
    ax.plot(time, state.history['thrust'], 'orange', linewidth=2, label='Thrust')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (kN)')
    ax.set_title('Thrust vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Burn rate
    ax = axes[0, 2]
    ax.plot(time, state.history['burn_rate'], 'gold', linewidth=2, label='Burn Rate')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Burn Rate (mm/s)')
    ax.set_title('Propellant Regression Rate')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # QRR Coherence
    ax = axes[1, 0]
    ax.plot(time, state.history['coherence'], 'cyan', linewidth=2, label='QRR Coherence')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Coherence')
    ax.set_title('QRR Relational Coherence')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Relational Energy
    ax = axes[1, 1]
    ax.plot(time, state.history['relational_energy'], 'magenta', linewidth=2, 
            label='Relational Energy')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('E_R (normalized)')
    ax.set_title('QRR Relational Energy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Grain geometry evolution
    ax = axes[1, 2]
    ax.plot(time, state.history['port_radius'], 'purple', linewidth=2, label='Port Radius')
    ax.axhline(y=state.geom.outer_radius*100, color='r', linestyle='--', 
               alpha=0.5, label='Outer Radius')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Radius (cm)')
    ax.set_title('Grain Geometry Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Use OUTPUT_DIR if no save_path specified
    if save_path is None:
        save_path = OUTPUT_DIR / 'qrr_rocket_motor_results.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()

def print_summary_statistics(state: QRRMotorState):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SIMULATION SUMMARY STATISTICS")
    print("="*70)
    
    pressure = np.array(state.history['pressure'])
    thrust = np.array(state.history['thrust'])
    coherence = np.array(state.history['coherence'])
    
    print(f"\nPressure Statistics:")
    print(f"  Peak pressure:     {np.max(pressure):.2f} MPa")
    print(f"  Average pressure:  {np.mean(pressure):.2f} MPa")
    print(f"  Pressure stability: {1 - np.std(pressure)/np.mean(pressure):.3f}")
    
    print(f"\nThrust Statistics:")
    print(f"  Peak thrust:       {np.max(thrust):.2f} kN")
    print(f"  Average thrust:    {np.mean(thrust):.2f} kN")
    print(f"  Total impulse:     {np.trapz(thrust*1000, state.history['time']):.0f} N·s")
    
    print(f"\nBurn Characteristics:")
    print(f"  Burn time:         {state.history['time'][-1]:.3f} s")
    print(f"  Final port radius: {state.history['port_radius'][-1]:.2f} cm")
    
    print(f"\nQRR Metrics:")
    print(f"  Average coherence:     {np.mean(coherence):.4f}")
    print(f"  Coherence stability:   {1 - np.std(coherence):.4f}")
    print(f"  Min coherence:         {np.min(coherence):.4f}")
    print(f"  Peak relational energy: {np.max(state.history['relational_energy']):.4f}")
    
    print("="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run QRR rocket motor simulation"""
    
    print("\n" + "="*70)
    print("QRR SOLID ROCKET MOTOR INTERNAL BALLISTICS SIMULATOR")
    print("Relational Relativity LLC")
    print("="*70 + "\n")
    
    # Initialize physics
    constants = PhysicalConstants()
    propellant = PropellantProperties()
    geometry = MotorGeometry()
    
    # Create motor state with QRR
    motor_state = QRRMotorState(constants, propellant, geometry)
    
    # Create simulator
    simulator = RocketMotorSimulator(motor_state, dt=0.001)
    
    # Run simulation
    simulator.run_simulation(max_time=10.0, record_interval=0.01)
    
    # Analysis
    print_summary_statistics(motor_state)
    
    # Visualization
    plot_results(motor_state)
    
    # Export data to OUTPUT_DIR
    data_path = OUTPUT_DIR / 'qrr_rocket_motor_data.npz'
    np.savez(data_path,
             time=motor_state.history['time'],
             pressure=motor_state.history['pressure'],
             thrust=motor_state.history['thrust'],
             burn_rate=motor_state.history['burn_rate'],
             coherence=motor_state.history['coherence'],
             relational_energy=motor_state.history['relational_energy'])
    
    print(f"Data exported to '{data_path}'")
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
