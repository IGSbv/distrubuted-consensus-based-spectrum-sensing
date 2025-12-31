import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from dist_phase1_topology import generate_network, get_faded_signals
from dist_phase2_hmm import train_node_hmms, get_initial_llrs
from dist_phase3_gossip import run_gossip_consensus

# --- EXECUTION ---
coords, adj = generate_network()
true_states, energies = get_faded_signals(2000, snr_db=-5)

train_energies = energies[:1000]
test_energies = energies[1000:]
test_labels = true_states[1000:]

models = train_node_hmms(train_energies)
initial_llrs = get_initial_llrs(models, test_energies)

results = {'Single Node': [], 'Optimized Gossip': [], 'Centralized (Ideal)': []}
HYSTERESIS_BUFFER = 0.5 
total_iters = 0

for t in range(1000):
    local_ops = initial_llrs[t, :]
    
    # Single Node
    results['Single Node'].append(1 if local_ops[0] > 0 else 0)
    
    # Optimized Gossip with Stability Checks
    final_vals, _, iters_used = run_gossip_consensus(local_ops, adj, momentum=0.2)
    
    # Hysteresis Decision Rule
    # Decision is 1 ONLY if consensus is clearly positive.
    avg_llr = np.mean(final_vals)
    results['Optimized Gossip'].append(1 if avg_llr > HYSTERESIS_BUFFER else 0)
    
    total_iters += iters_used
    results['Centralized (Ideal)'].append(1 if np.sum(local_ops) > 0 else 0)

# --- RESULTS OUTPUT ---
print("\n" + "="*40)
print("STABILIZED DISTRIBUTED SENSING RESULTS")
print("="*40)
for method, decisions in results.items():
    acc = accuracy_score(test_labels, decisions)
    print(f"{method:20}: {acc*100:.2f}%")

avg_energy_saved = (1 - (total_iters / (1000 * 30))) * 100
print(f"Total Gossip Iterations: {total_iters}")
print(f"\nAverage Network Energy Saved: {avg_energy_saved:.2f}%")