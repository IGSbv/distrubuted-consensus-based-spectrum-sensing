# dist_phase4_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Import logic from previous phases (Assuming they are in the same directory)
from dist_phase1_topology import generate_network, get_faded_signals
from dist_phase2_hmm import train_node_hmms, get_initial_llrs
from dist_phase3_gossip import run_gossip_consensus

# --- EXECUTION ---
# 1. Setup Environment
coords, adj = generate_network()
true_states, energies = get_faded_signals(2000, snr_db=-5) # 2000 samples for robust testing

# 2. Local Intelligence
# Split into Train (1000) and Test (1000)
train_energies = energies[:1000]
test_energies = energies[1000:]
test_labels = true_states[1000:]

models = train_node_hmms(train_energies)
initial_llrs = get_initial_llrs(models, test_energies)

# 3. Performance Metrics
results = {
    'Single Node': [],
    'Gossip (5 iter)': [],
    'Gossip (20 iter)': [],
    'Centralized (Ideal)': []
}

# Process each time sample
for t in range(1000):
    local_opinions = initial_llrs[t, :]
    
    # a. Single Node (User 0)
    results['Single Node'].append(1 if local_opinions[0] > 0 else 0)
    
    # b. Distributed Gossip (5 iterations)
    final_5, _ = run_gossip_consensus(local_opinions, adj, iterations=5)
    results['Gossip (5 iter)'].append(1 if final_5[0] > 0 else 0)
    
    # c. Distributed Gossip (20 iterations)
    final_20, _ = run_gossip_consensus(local_opinions, adj, iterations=20)
    results['Gossip (20 iter)'].append(1 if final_20[0] > 0 else 0)
    
    # d. Centralized Ideal (The standard sum of all LLRs)
    results['Centralized (Ideal)'].append(1 if np.sum(local_opinions) > 0 else 0)

# 4. Calculate Final Accuracies
print("\n" + "="*30)
print("FINAL ACCURACY RESULTS")
print("="*30)
for method, decisions in results.items():
    acc = accuracy_score(test_labels, decisions)
    print(f"{method:20}: {acc*100:.2f}%")

# 5. Visualization: Convergence of a single time sample
t_sample = np.random.randint(0, 1000)
_, history = run_gossip_consensus(initial_llrs[t_sample, :], adj, iterations=30)

plt.figure(figsize=(10, 5))
plt.plot(history)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.title(f"LLR Consensus Convergence (Time Step {t_sample})")
plt.xlabel("Gossip Iterations")
plt.ylabel("Log-Likelihood Ratio (LLR)")
plt.legend([f"Node {i}" for i in range(10)], loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()