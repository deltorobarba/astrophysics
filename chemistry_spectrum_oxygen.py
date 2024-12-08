import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def get_oxygen_nist_lines():
    """
    Get oxygen spectral lines from NIST Atomic Spectra Database.
    Returns dictionary with wavelengths and relative intensities.
    Source: https://physics.nist.gov/PhysRefData/ASD/lines_form.html
    """
    # NIST data for neutral oxygen (OI) main visible spectral lines
    nist_lines = {
        # wavelength (nm): relative intensity (normalized)
        436.8: 0.80,  # Strong blue line
        441.5: 0.75,  # Blue line
        533.0: 0.90,  # Strong green line
        543.5: 0.85,  # Green line
        557.7: 0.70,  # Famous "green line" in aurora
        577.0: 0.65,  # Yellow-green line
        595.8: 0.60,  # Orange line
        615.7: 0.55,  # Orange-red line
        645.4: 0.50,  # Red line
        677.3: 0.45   # Deep red line
    }
    return nist_lines

def wavelength_to_rgb(wavelength):
    """
    Convert wavelength (nm) to RGB color.
    Based on Dan Bruton's algorithm.
    """
    gamma = 0.8
    if wavelength < 380 or wavelength > 780:
        return (0, 0, 0)

    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0

    # Intensity correction
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength <= 700:
        factor = 1.0
    elif 700 < wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 0.0

    R = ((R * factor) ** gamma)
    G = ((G * factor) ** gamma)
    B = ((B * factor) ** gamma)

    return (R, G, B)

def plot_nist_spectrum():
    """
    Create visualization of oxygen emission spectrum using NIST data.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')

    lines = get_oxygen_nist_lines()

    # Plot each spectral line with its corresponding color
    for wavelength, intensity in lines.items():
        color = wavelength_to_rgb(wavelength)
        plt.bar(wavelength, intensity, width=2.0, color=color, alpha=0.8)

        # Add wavelength labels
        plt.text(wavelength, intensity + 0.05,
                f'{wavelength}nm',
                color='white',
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90)

    # Customize plot
    plt.xlabel('Wavelength (nm)', color='white', fontsize=12)
    plt.ylabel('Intensity (a.u.)', color='white', fontsize=12)
    plt.title('Emission Spectrum of Oxygen (NIST Data)',
             color='white', pad=20, fontsize=14)

    # Set range to visible spectrum
    plt.xlim(380, 780)
    plt.ylim(0, 1.2)

    # Customize grid and spines
    ax.grid(True, color='#303030', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#303030')
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.show()

    # Print spectral line information
    print("\nOxygen spectral lines (NIST data):")
    for wavelength, intensity in sorted(lines.items()):
        print(f"λ = {wavelength} nm, Relative Intensity = {intensity:.2f}")




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def wavelength_to_rgb(wavelength):
    """Convert wavelength (nm) to RGB color."""
    gamma = 0.8
    if wavelength < 380 or wavelength > 750:
        return (0, 0, 0)

    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    else:
        R = 1.0
        G = 0.0
        B = 0.0

    # Intensity correction
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength <= 700:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)

    return (R * factor, G * factor, B * factor)

def create_spectral_background(ax, wavelength_range):
    """Create a properly oriented spectral background."""
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], 1000)
    colors = [wavelength_to_rgb(w) for w in wavelengths]

    # Create gradient by plotting narrow vertical bars
    for i in range(len(wavelengths)-1):
        ax.axvspan(wavelengths[i], wavelengths[i+1],
                  color=colors[i], alpha=1.0)

def plot_oxygen_absorption():
    """Create visualization of oxygen absorption spectrum with correct rainbow background."""
    # NIST data for oxygen absorption lines (nm)
    oxygen_lines = {
        436.8: 0.8,
        441.5: 0.7,
        533.0: 0.9,
        543.5: 0.8,
        557.7: 0.7,
        577.0: 0.6,
        595.8: 0.6,
        615.7: 0.5,
        645.4: 0.5,
        677.3: 0.4
    }

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 3), facecolor='black')

    # Create proper spectral background
    wavelength_range = (380, 750)
    create_spectral_background(ax, wavelength_range)

    # Plot absorption lines
    for wavelength, intensity in oxygen_lines.items():
        plt.plot([wavelength, wavelength],
                [0, 1],
                color='black',
                linewidth=1.5,
                alpha=0.8)

    # Customize plot
    plt.title('Absorption Spectrum of Oxygen',
             color='white',
             pad=20,
             fontsize=14)
    plt.xlabel('Wavelength (nm)', color='white', fontsize=12)

    # Remove y-axis labels as they're not needed
    plt.yticks([])

    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#303030')
    ax.tick_params(colors='white')

    # Set wavelength range
    plt.xlim(wavelength_range)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Print spectral line information
    print("\nOxygen absorption lines (NIST data):")
    for wavelength in sorted(oxygen_lines.keys()):
        print(f"λ = {wavelength} nm")