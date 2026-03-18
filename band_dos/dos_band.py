import matplotlib.pyplot as plt
import numpy as np
from pymatgen.io.vasp import Vasprun

# Simple version focusing on essential plots
print("Loading vasprun.xml...")
vr = Vasprun("siband2_vasprun.xml")

# Basic DOS plot
print("Plotting basic DOS...")
dos = vr.complete_dos
energies = dos.energies - dos.efermi
total_dos = dos.get_densities()

plt.figure(figsize=(10, 6))
plt.plot(total_dos, energies, 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', label='Fermi level')
plt.xlabel('DOS (states/eV)')
plt.ylabel('Energy - E_F (eV)')
plt.title('Si Density of States')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, None)
plt.ylim(-13, 13)
plt.tight_layout()
# plt.savefig('si_dos_simple.png', dpi=300)
plt.show()

# Basic band structure if available
try:
    print("Plotting band structure...")
    bs = vr.get_band_structure()
    
    # Simple band structure plot
    plt.figure(figsize=(12, 8))
    for spin in bs.bands:
        for band_idx in range(len(bs.bands[spin])):
            plt.plot(bs.bands[spin][band_idx] - bs.efermi, 'b-', alpha=0.5)
    
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Fermi level')
    plt.xlabel('k-points')
    plt.ylabel('Energy - E_F (eV)')
    plt.title('Si Band Structure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-15, 15)
    plt.tight_layout()
    # plt.savefig('si_bands_simple.png', dpi=300)
    plt.show()
    
except Exception as e:
    print(f"Could not plot band structure: {e}")

print(f"Fermi energy: {dos.efermi:.4f} eV")
print(f"Final energy: {vr.final_energy:.6f} eV")

