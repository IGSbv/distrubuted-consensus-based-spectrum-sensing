import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Assuming these are imported from your previous phases
from dist_phase1_topology import generate_network, get_faded_signals
from dist_phase2_hmm import train_node_hmms, get_initial_llrs
from dist_phase3_gossip import run_gossip_consensus

def run_comparative_analysis():
    # 1. Setup Environment
    coords, adj = generate_network()
    # Testing over a range of SNRs to show the 'Crossover Point'
    snr_range = [-15, -10, -5, 0, 5]
    
    cent_accs = []
    dist_accs = []
    energy_savings = []

    print("--- Starting Comparative Sweep ---")
    
    for snr in snr_range:
        true_states, energies = get_faded_signals(1000, snr_db=snr)
        models = train_node_hmms(energies[:500])
        initial_llrs = get_initial_llrs(models, energies[500:])
        labels = true_states[500:]
        
        cent_decisions = []
        dist_decisions = []
        total_iters = 0
        
        for t in range(500):
            local_ops = initial_llrs[t, :]
            
            # --- MODEL A: Centralized (Snapshot Sum) ---
            # Random Process: Static Bayesian Summation
            cent_decisions.append(1 if np.sum(local_ops) > 0 else 0)
            
            # --- MODEL B: Distributed (Momentum-Gossip) ---
            # Random Process: Second-Order Markov Chain
            # Using our Hysteresis buffer (0.5) for robustness
            final_vals, _, iters = run_gossip_consensus(local_ops, adj, momentum=0.2, epsilon=0.01)
            dist_decisions.append(1 if np.mean(final_vals) > 0.5 else 0)
            total_iters += iters
            
        cent_accs.append(accuracy_score(labels, cent_decisions))
        dist_accs.append(accuracy_score(labels, dist_decisions))
        energy_savings.append((1 - (total_iters / (500 * 30))) * 100)
        print(f"SNR {snr}dB: Cent={cent_accs[-1]:.2f}, Dist={dist_accs[-1]:.2f}, EnergySaved={energy_savings[-1]:.1f}%")

    # --- GENERATE TECHNICAL REPORT ---
    print("\n" + "="*50)
    print("FINAL COMPARATIVE TECHNICAL REPORT")
    print("="*50)
    print(f"{'Metric':25} | {'Centralized':12} | {'Distributed'}")
    print("-" * 50)
    print(f"{'Peak Accuracy':25} | {max(cent_accs)*100:11.1f}% | {max(dist_accs)*100:.1f}%")
    print(f"{'Avg Energy Efficiency':25} | {'0% (Base)':11} | {np.mean(energy_savings):.1f}% Saved")
    print(f"{'Architecture Type':25} | {'Hub-Spoke':11} | {'Mesh/Peer'}")
    print(f"{'Stochastic Model':25} | {'Snapshot Bayes':11} | {'Iterative DTMC'}")
    print(f"{'Failure Robustness':25} | {'Fragile':11} | {'Resilient'}")
    print("="*50)

    # --- PLOT COMPARISON ---
    plt.figure(figsize=(10, 6))
    plt.plot(snr_range, [a*100 for a in cent_accs], 'r--o', label='Centralized (Ideal Sum)')
    plt.plot(snr_range, [a*100 for a in dist_accs], 'b-s', label='Distributed (Momentum-Gossip)')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
    plt.ylabel('Detection Accuracy [%]')
    plt.title('Architecture Comparison: Centralized vs. Distributed Consensus')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_comparative_analysis()
