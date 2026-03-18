import numpy as np
import matplotlib.pyplot as plt

def read_qe_freq_with_qvectors(filename):
    q_vectors = []
    freqs = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("&plot") or not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) == 4:
            q_vectors.append([float(parts[1]),
                              float(parts[2]),
                              float(parts[3])])
            i += 1
            freqs.append(list(map(float, lines[i].split())))
        i += 1

    return np.array(q_vectors), np.array(freqs)


def cumulative_distance(q):
    dist = np.zeros(len(q))
    for i in range(1, len(q)):
        dist[i] = dist[i-1] + np.linalg.norm(q[i] - q[i-1])
    return dist


def plot_phonon_dispersion_correct(
    filename,
    sym_indices,
    sym_labels,
    unit="cm-1"
):
    qvecs, freqs = read_qe_freq_with_qvectors(filename)
    x = cumulative_distance(qvecs)

    if unit.lower() == "thz":
        freqs *= 0.0299792458
        ylabel = "Frequency (THz)"
    else:
        ylabel = "Frequency (cm$^{-1}$)"

    plt.figure(figsize=(8, 5))

    for i in range(freqs.shape[1]):
        plt.plot(x, freqs[:, i], color="black", linewidth=1)

    sym_x = [x[i] for i in sym_indices]

    for s in sym_x:
        plt.axvline(s, color="gray", linestyle="--", linewidth=0.8)

    plt.xticks(sym_x, sym_labels)
    plt.xlabel("Wave vector path")
    plt.ylabel(ylabel)
    plt.title("Phonon Dispersion")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ======================
# Example for your path:
# Γ → X → K → Γ → L
# ======================

# Indices where each segment starts (must match matdyn input)
sym_indices = [0, 30, 40, 60, 90]
sym_labels = [r"$\Gamma$", "X", "K", r"$\Gamma$", "L"]

plot_phonon_dispersion_correct(
    "si.freq",
    sym_indices,
    sym_labels
)
