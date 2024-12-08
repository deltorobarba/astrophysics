import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def get_atomic_levels(element_name: str) -> Dict[str, float]:
    """
    Get atomic energy levels for elements.

    Args:
        element_name (str): Chemical symbol (e.g. 'Ca') or element name (e.g. 'Calcium')

    Returns:
        Dict[str, float]: Dictionary of atomic states and their energies in eV
    """
    try:
        # Convert full element name to symbol if needed
        element_name = element_name.capitalize()

        # Expanded database of energy levels (in eV)
        # Data sourced from NIST Atomic Spectra Database
        energy_database = {
            'Ca': {'4s': 0.0, '4p': 2.93, '3d': 1.88},
            'Na': {'3s': 0.0, '3p': 2.10},
            'Li': {'2s': 0.0, '2p': 1.85},
            'K':  {'4s': 0.0, '4p': 1.61},
            'Mg': {'3s': 0.0, '3p': 2.71},
            'He': {'1s': 0.0, '2s': 20.62, '2p': 21.22},
            'H':  {'1s': 0.0, '2s': 10.2, '2p': 10.2},
            'Be': {'2s': 0.0, '2p': 3.96},
            'B':  {'2p': 0.0, '3s': 3.64},
            'C':  {'2p': 0.0, '3s': 7.48, '3p': 8.64},
            'N':  {'2p': 0.0, '3s': 10.33, '3p': 11.44},
            'O':  {'2p': 0.0, '3s': 9.51, '3p': 10.74},
            'F':  {'2p': 0.0, '3s': 12.97, '3p': 14.03},
            'Ne': {'2p': 0.0, '3s': 16.62, '3p': 18.55},
            'Al': {'3p': 0.0, '4s': 3.14, '3d': 4.02},
            'Si': {'3p': 0.0, '4s': 4.92, '3d': 6.62},
            'P':  {'3p': 0.0, '4s': 6.95, '3d': 8.48},
            'S':  {'3p': 0.0, '4s': 8.42, '3d': 10.29},
            'Cl': {'3p': 0.0, '4s': 10.4, '3d': 11.65},
            'Ar': {'3p': 0.0, '4s': 11.72, '3d': 13.48},
            'Sc': {'4s': 0.0, '3d': 1.43, '4p': 3.12},
            'Ti': {'4s': 0.0, '3d': 0.81, '4p': 3.34},
            'V':  {'4s': 0.0, '3d': 0.26, '4p': 3.11},
            'Cr': {'4s': 0.0, '3d': 0.94, '4p': 3.32},
            'Mn': {'4s': 0.0, '3d': 2.14, '4p': 3.53},
            'Fe': {'4s': 0.0, '3d': 0.86, '4p': 3.69},
            'Co': {'4s': 0.0, '3d': 0.43, '4p': 3.88},
            'Ni': {'4s': 0.0, '3d': 0.03, '4p': 3.66},
            'Cu': {'4s': 0.0, '4p': 3.79, '3d': 1.39},
            'Zn': {'4s': 0.0, '4p': 5.80, '3d': 6.01},
            'Rb': {'5s': 0.0, '5p': 1.56, '4d': 2.40},
            'Sr': {'5s': 0.0, '5p': 2.69, '4d': 1.80},
            'Cs': {'6s': 0.0, '6p': 1.39, '5d': 2.71},
            'Ba': {'6s': 0.0, '6p': 2.24, '5d': 1.67},
            'Ga': {'4p': 0.0, '5s': 3.07, '4d': 4.31},
            'Ge': {'4p': 0.0, '5s': 4.96, '4d': 6.52},
            'As': {'4p': 0.0, '5s': 6.59, '4d': 8.11},
            'Se': {'4p': 0.0, '5s': 7.87, '4d': 9.58},
            'Br': {'4p': 0.0, '5s': 9.45, '4d': 11.12},
            'Kr': {'4p': 0.0, '5s': 10.56, '4d': 12.36},
            'In': {'5p': 0.0, '6s': 3.02, '5d': 4.08},
            'Sn': {'5p': 0.0, '6s': 4.87, '5d': 6.18},
            'Sb': {'5p': 0.0, '6s': 6.32, '5d': 7.84},
            'Te': {'5p': 0.0, '6s': 7.79, '5d': 9.51},
            'I':  {'5p': 0.0, '6s': 9.14, '5d': 10.92},
            'Xe': {'5p': 0.0, '6s': 10.56, '5d': 12.13}
        }

        if element_name in energy_database:
            return energy_database[element_name]
        else:
            print(f"Energy levels for {element_name} are not in the database.")
            print("Available elements:", ", ".join(sorted(energy_database.keys())))
            return {}

    except Exception as e:
        print(f"Error getting data for {element_name}: {e}")
        return {}

def energy_to_wavelength(energy_eV):
    """Convert energy in eV to wavelength in nm."""
    h = 4.1357e-15  # Planck's constant in eV·s
    c = 3e8  # Speed of light in m/s
    return (h * c) / energy_eV * 1e9  # Convert to nm

def spectral_analysis(atom_name, energy_levels):
    """
    Calculate and visualize the emission and absorption spectra for a given atom.
    
    Parameters:
    atom_name (str): Name of the atom
    energy_levels (dict): Dictionary containing energy levels in eV with state names as keys
    """
    # Set the style to dark background
    plt.style.use('dark_background')

    # Custom colors
    EMISSION_COLOR = '#ff9500'  # Bright orange
    ABSORPTION_COLOR = '#00b4d8'  # Bright blue
    BASELINE_COLOR = '#404040'  # Dark gray
    GRID_COLOR = '#303030'  # Slightly lighter gray for grid
    
    # Calculate transitions
    transitions = {}
    states = list(energy_levels.keys())
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            state1, state2 = states[j], states[i]
            energy_diff = energy_levels[state1] - energy_levels[state2]
            transitions[(state1, state2)] = energy_diff
    
    # Convert to wavelengths
    wavelengths = {transition: abs(energy_to_wavelength(energy)) 
                  for transition, energy in transitions.items()}
    
    # Create figure with dark background
    plt.figure(figsize=(12, 8), facecolor='black')
    
    # Emission Spectrum
    plt.subplot(2, 1, 1)
    plt.bar(list(wavelengths.values()), 
            height=[1.0] * len(wavelengths), 
            width=5.0, 
            color=EMISSION_COLOR, 
            alpha=0.8)
    
    plt.xlabel('Wavelength (nm)', color='white', fontsize=10)
    plt.ylabel('Intensity (a.u.)', color='white', fontsize=10)
    plt.title(f'Emission Spectrum of {atom_name} (Electronic Transitions)', 
             color='white', pad=20, fontsize=12)
    
    # Customize grid and spines
    plt.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.5)
    for spine in plt.gca().spines.values():
        spine.set_color(GRID_COLOR)
    plt.tick_params(colors='white')
    
    # Absorption Spectrum
    plt.subplot(2, 1, 2)
    
    # Calculate appropriate wavelength range
    min_wavelength = min(wavelengths.values()) * 0.8
    max_wavelength = max(wavelengths.values()) * 1.2
    
    # Create baseline
    plt.plot(np.linspace(min_wavelength, max_wavelength, 1000), 
             np.ones(1000), 
             color=BASELINE_COLOR, 
             linewidth=1)
    
    # Plot absorption lines with gradient effect
    for transition, wavelength in wavelengths.items():
        y = np.linspace(0, 1, 100)
        x = np.ones_like(y) * wavelength
        plt.plot(x, y, color=ABSORPTION_COLOR, alpha=0.8, linewidth=2)
        
        # Add transition labels
        plt.text(wavelength, 1.1, 
                f"{transition[0]}→{transition[1]}\n{wavelength:.0f} nm",
                color='white', ha='center', va='bottom', fontsize=8,
                rotation=90)
    
    plt.xlabel('Wavelength (nm)', color='white', fontsize=10)
    plt.ylabel('Normalized Intensity (a.u.)', color='white', fontsize=10)
    plt.title(f'Absorption Spectrum of {atom_name} (Electronic Transitions)', 
             color='white', pad=20, fontsize=12)
    
    # Set wavelength range
    plt.xlim(min_wavelength, max_wavelength)
    
    # Customize grid and spines
    plt.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.5)
    for spine in plt.gca().spines.values():
        spine.set_color(GRID_COLOR)
    plt.tick_params(colors='white')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    
    # Set figure background to black
    plt.gcf().patch.set_facecolor('black')
    
    plt.show()
    
    # Print transition information
    print(f"\nElectronic transitions for {atom_name}:")
    for transition, wavelength in sorted(wavelengths.items(), key=lambda x: x[1]):
        print(f"{transition[0]} → {transition[1]}: {wavelength:.2f} nm")

def format_energy_levels(energy_levels: Dict[str, float]) -> None:
    """Print energy levels in a nice format with comments."""
    if not energy_levels:
        return

    print("\nEnergy levels:")
    for state, energy in sorted(energy_levels.items(), key=lambda x: x[1]):
        if energy == 0.0:
            print(f"'{state}': {energy:.2f},  # Ground state")
        else:
            print(f"'{state}': {energy:.2f},  # Excited state")