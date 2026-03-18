import numpy as np
import matplotlib.pyplot as plt

def read_phdos(filename):
    """
    Reads a Quantum ESPRESSO phonon DOS (phdos) file.

    Returns:
        freq : 1D array
        dos  : 1D array (total DOS)
        pdos : 2D array (PDOS per atom, shape: [n_atoms, n_points])
    """
    data = np.loadtxt(filename, comments="#")

    freq = data[:, 0]
    dos = data[:, 1]

    if data.shape[1] > 2:
        pdos = data[:, 2:].T
    else:
        pdos = None

    return freq, dos, pdos


def plot_phdos(filename, unit="cm-1"):
    freq, dos, pdos = read_phdos(filename)

    # Unit conversion
    if unit.lower() == "thz":
        freq = freq * 0.0299792458
        xlabel = "Frequency (THz)"
    else:
        xlabel = "Frequency (cm$^{-1}$)"

    plt.figure(figsize=(7, 5))

    # Total DOS
    plt.plot(freq, dos, label="Total DOS", linewidth=2)

    # Projected DOS (if present)
    if pdos is not None:
        for i, p in enumerate(pdos, start=1):
            plt.plot(freq, p, "--", label=f"PDOS atom {i}")

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Phonon DOS (states / unit frequency)", fontsize=12)
    plt.title("Phonon Density of States", fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_phdos("si.phdos", unit="cm-1")
# plot_phdos("si.phdos", unit="thz")
