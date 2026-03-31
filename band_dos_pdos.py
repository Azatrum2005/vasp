import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
import numpy as np

# Load the vasprun.xml file
vasprun = Vasprun("vasprun.xml")

# 1. Plot Band Structure
print("Plotting band structure...")
bandstructure = vasprun.get_band_structure()
bs_plotter = BSPlotter(bandstructure)
plt.figure(figsize=(10, 8))
bs_plotter.get_plot(ylim=(-15, 15))
plt.title("Si Band Structure")
plt.tight_layout()
plt.savefig("band_dos/band_structure.png", dpi=300)
plt.show()

# 2. Plot Density of States (DOS)
print("Plotting density of states...")
dos = vasprun.complete_dos
dos_plotter = DosPlotter()
dos_plotter.add_dos("Total DOS", dos)
plt.figure(figsize=(10, 6))
dos_plotter.get_plot(xlim=(-15, 15), ylim=(0, None))
plt.title("Density of States")
plt.tight_layout()
plt.savefig("band_dos/density_of_states.png", dpi=300)
plt.show()

# 3. Plot Partial Density of States (PDOS) - Fixed version
print("Plotting partial density of states...")
pdos_plotter = DosPlotter()

# Check what orbital projections are available
print("Checking available orbital projections...")
available_orbitals = set()

for site in dos.structure:
    try:
        site_dos = dos.get_site_spd_dos(site)
        available_orbitals.update(site_dos.keys())
        print(f"Site {site.species_string}: Available orbitals: {list(site_dos.keys())}")
    except Exception as e:
        print(f"Could not get orbitals for site {site.species_string}: {e}")

# Try element-based PDOS instead
print("\nTrying element-based PDOS...")
for element in set([site.species_string for site in dos.structure]):
    try:
        element_dos = dos.get_element_dos(element)
        pdos_plotter.add_dos(f"{element} Total", element_dos)
        print(f"Added {element} total DOS")
    except Exception as e:
        print(f"Could not get element DOS for {element}: {e}")

# Try to get orbital projections by element
for element in set([site.species_string for site in dos.structure]):
    try:
        element_spd = dos.get_element_spd_dos(element)
        for orbital, orbital_dos in element_spd.items():
            pdos_plotter.add_dos(f"{element} {orbital}", orbital_dos)
            print(f"Added {element} {orbital} DOS")
    except Exception as e:
        print(f"Could not get orbital projections for {element}: {e}")

plt.figure(figsize=(12, 8))
pdos_plotter.get_plot(xlim=(-15, 15), ylim=(0, None))
plt.title("Partial Density of States")
plt.tight_layout()
plt.savefig("band_dos/pdos.png", dpi=300)
plt.show()

# 4. Get calculation information
print("\n=== Calculation Information ===")
print(f"System: {vasprun.parameters['SYSTEM']}")
print(f"Final energy: {vasprun.final_energy:.6f} eV")
print(f"Fermi energy: {dos.efermi:.4f} eV")
print(f"Number of atoms: {len(dos.structure)}")
print(f"Volume: {dos.structure.volume:.2f} Å³")

# 5. Band gap analysis
print("\n=== Band Gap Analysis ===")
band_gap = bandstructure.get_band_gap()
print(band_gap)
print(f"Band gap: {band_gap['energy']:.4f} eV")
print(f"Direct gap: {band_gap['direct']}")
if 'transition' in band_gap:
    print(f"Transition: {band_gap['transition']}")

dos_data = dos.get_densities()
energies = dos.energies - dos.efermi

# 6. Convergence data
print("Plotting energy convergence...")
if hasattr(vasprun, 'ionic_steps') and vasprun.ionic_steps:
    energies = [step['e_0_energy'] for step in vasprun.ionic_steps]
    plt.figure(figsize=(10, 6))
    plt.plot(energies, 'bo-', linewidth=2)
    plt.xlabel('SCF Step')
    plt.ylabel('Energy (eV)')
    plt.title('Energy Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("band_dos/convergence.png", dpi=300)
    plt.show()
