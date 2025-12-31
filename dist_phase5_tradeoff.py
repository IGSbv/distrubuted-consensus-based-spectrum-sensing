# dist_phase5_tradeoff.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Import logic from previous phases
from dist_phase1_topology import generate_network, get_faded_signals
from dist_phase2_hmm import train_node_hmms, get_initial_llrs
from dist_phase3_gossip import run_gossip_consensus

def run_performance_sweep():
    # 1. Setup Environment & Fixed Topology
    coords, adj = generate_network()
    snr_range = [-15, -10, -5, 0, 5]
    iteration_counts = [0, 1, 3, 5, 10, 20]
    
    results = np.zeros((len(snr_range), len(iteration_counts)))
    
    print("--- Starting Distributed Tradeoff Sweep ---")
    
    for i, snr in enumerate(snr_range):
        print(f"Testing SNR = {snr} dB...")
        
        # Generate data for this specific SNR
        true_states, energies = get_faded_signals(1500, snr_db=snr)
        
        # Local Intelligence (Train on first 500, Test on next 1000)
        models = train_node_hmms(energies[:500])
        initial_llrs = get_initial_llrs(models, energies[500:])
        test_labels = true_states[500:]
        
        for j, iters in enumerate(iteration_counts):
            # Process all 1000 samples for this iteration count
            batch_decisions = []
            for t in range(1000):
                # Run gossip
                final_llr, _ = run_gossip_consensus(initial_llrs[t, :], adj, iterations=iters)
                # Decision for Node 0 (representative of the mesh)
                batch_decisions.append(1 if final_llr[0] > 0 else 0)
            
            results[i, j] = accuracy_score(test_labels, batch_decisions)
            
    return snr_range, iteration_counts, results

if __name__ == "__main__":
    snr_axis, iter_axis, data = run_performance_sweep()
    
    # Visualization: Accuracy vs Iterations for different SNRs
    plt.figure(figsize=(10, 6))
    for i in range(len(snr_axis)):
        plt.plot(iter_axis, data[i, :] * 100, '-o', label=f'SNR = {snr_axis[i]} dB')
    
    plt.title("The Saturation Point: Accuracy vs. Gossip Iterations")
    plt.xlabel("Gossip Iterations (Communication Cost)")
    plt.ylabel("Detection Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(50, 100)
    plt.show()
    
    print("Phase 5 Complete: Optimization Tradeoff Map Generated.")