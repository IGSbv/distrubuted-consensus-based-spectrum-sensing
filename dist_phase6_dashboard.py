# dist_phase6_dashboard.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Import modules from your local project files
try:
    from dist_phase1_topology import generate_network, get_faded_signals
    from dist_phase2_hmm import train_node_hmms, get_initial_llrs
    from dist_phase3_gossip import run_gossip_consensus
except ImportError:
    print("ERROR: Make sure dist_phase1_topology.py, dist_phase2_hmm.py, and dist_phase3_gossip.py are in this folder.")

# --- CONFIGURATION ---
WINDOW_SIZE = 100
UPDATE_INTERVAL_MS = 100 
GOSSIP_ITER_PER_FRAME = 5

# --- INITIALIZATION ---
def init_system():
    print("--- Dashboard Startup: Preparing Distributed Mesh ---")
    coords, adj = generate_network()
    # Pre-train models with a small batch
    _, train_energies = get_faded_signals(500, snr_db=-5)
    models = train_node_hmms(train_energies)
    print("--- System Ready: Starting Live Mesh Stream ---")
    return coords, adj, models

coords, adj, trained_models = init_system()

# GLOBAL BUFFERS (Initialized with zeros to prevent length-mismatch errors)
true_state_buffer = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
node0_llr_buffer = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
consensus_llr_buffer = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)

def update_plot(frame):
    # 1. Generate one new live time-step
    true_state = 1 if (frame // 20) % 2 == 0 else 0
    
    live_energies = np.zeros((1, 10))
    for u in range(10):
        h = (np.random.normal(0, 1) + 1j*np.random.normal(0, 1))/np.sqrt(2)
        noise_pow = 10**(5/10) # SNR = -5dB (10^(0.5))
        noise = (np.random.normal(0, np.sqrt(noise_pow/2)) + 1j*np.random.normal(0, np.sqrt(noise_pow/2)))
        live_energies[0, u] = np.abs(h * true_state + noise)**2

    # 2. Local HMM Inference
    current_llrs = get_initial_llrs(trained_models, live_energies)[0]
    
    # 3. Distributed Gossip
    final_consensus, _ = run_gossip_consensus(current_llrs, adj, iterations=GOSSIP_ITER_PER_FRAME)
    
    # 4. Update Buffers
    true_state_buffer.append(true_state)
    node0_llr_buffer.append(current_llrs[0])
    consensus_llr_buffer.append(final_consensus[0])
    
    # 5. Update Lines
    line_true.set_ydata(list(true_state_buffer))
    line_local.set_ydata(list(node0_llr_buffer))
    line_consensus.set_ydata(list(consensus_llr_buffer))
    
    return line_true, line_local, line_consensus

# --- PLOT SETUP ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Subplot 1: Truth
ax1.set_title("Distributed Sensing Real-Time Demo (SNR = -5dB)")
ax1.set_ylim(-0.2, 1.2)
ax1.set_ylabel("True PU State")
line_true, = ax1.plot(range(WINDOW_SIZE), list(true_state_buffer), color='green', linewidth=2)

# Subplot 2: LLRs
ax2.set_ylim(-15, 15)
ax2.set_ylabel("Log-Likelihood Ratio")
ax2.set_xlabel("Time (Scrolling Window)")
line_local, = ax2.plot(range(WINDOW_SIZE), list(node0_llr_buffer), color='purple', alpha=0.4, label="Node 0 (Local)")
line_consensus, = ax2.plot(range(WINDOW_SIZE), list(consensus_llr_buffer), color='blue', linewidth=2, label="Network Consensus")
ax2.axhline(0, color='black', linestyle='--')
ax2.legend(loc='upper right')

ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL_MS, blit=False)
plt.tight_layout()
plt.show()