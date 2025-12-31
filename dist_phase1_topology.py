# dist_phase1_topology.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- CONFIGURATION ---
NUM_USERS = 10
FIELD_SIZE = 100        # 100x100 meter area
COMM_RADIUS = 60        # Nodes within 40m can communicate
NUM_SAMPLES = 1000
SNR_dB = -5

def generate_network():
    """Places nodes randomly and creates the Adjacency Matrix."""
    # 1. Random Node Placement
    coords = np.random.rand(NUM_USERS, 2) * FIELD_SIZE
    
    # 2. Create Adjacency Matrix (Distance-based)
    dist_matrix = cdist(coords, coords)
    adj_matrix = (dist_matrix < COMM_RADIUS).astype(int)
    np.fill_diagonal(adj_matrix, 0) # No self-loops
    
    return coords, adj_matrix

def get_faded_signals(n_samples, snr_db):
    """Generates PU activity and Rayleigh faded signals for all nodes."""
    # PU Activity (Ground Truth)
    true_states = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    tx_signal = true_states * np.random.choice([-1, 1], size=n_samples)
    
    rx_energies = np.zeros((n_samples, NUM_USERS))
    
    for u in range(NUM_USERS):
        # Rayleigh Fading + AWGN
        h = (np.random.normal(0, 1, n_samples) + 1j*np.random.normal(0, 1, n_samples))/np.sqrt(2)
        noise_pow = 10**(-snr_db/10)
        noise = (np.random.normal(0, np.sqrt(noise_pow/2), n_samples) + 
                 1j*np.random.normal(0, np.sqrt(noise_pow/2), n_samples))
        
        rx_signal = h * tx_signal + noise
        rx_energies[:, u] = np.abs(rx_signal)**2
        
    return true_states, rx_energies

if __name__ == "__main__":
    coords, adj = generate_network()
    true_states, energies = get_faded_signals(NUM_SAMPLES, SNR_dB)
    
    # Visualize Topology
    plt.figure(figsize=(6, 6))
    for i in range(NUM_USERS):
        for j in range(i + 1, NUM_USERS):
            if adj[i, j] == 1:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'k--', alpha=0.3)
    plt.scatter(coords[:,0], coords[:,1], c='blue', s=100, label='Secondary Users')
    plt.title("Distributed Network Topology (Ad-Hoc Mesh)")
    plt.legend()
    plt.show()
    print("Phase 1 Complete: Network Topology and Fading Environment Generated.")