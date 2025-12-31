# dist_phase3_gossip.py
import numpy as np

def run_gossip_consensus(initial_llrs, adj_matrix, iterations=20):
    """
    Simulates the iterative exchange of LLRs between neighbors.
    - initial_llrs: [num_users] (The HMM output for a single time step)
    - adj_matrix: [num_users x num_users] (From Phase 1)
    """
    num_users = len(initial_llrs)
    # Calculate degrees (how many neighbors each node has)
    degrees = np.sum(adj_matrix, axis=1)
    
    # x holds the current LLR estimate for each node
    x = np.copy(initial_llrs).astype(float)
    
    # Record history to visualize convergence later
    history = [np.copy(x)]
    
    for k in range(iterations):
        x_next = np.copy(x)
        for i in range(num_users):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            for j in neighbors:
                # Metropolis-Hastings Weighting
                # This ensures the 'Random Walk' on the graph is doubly stochastic
                weight = 1 / (max(degrees[i], degrees[j]) + 1)
                x_next[i] += weight * (x[j] - x[i])
        
        x = x_next
        history.append(np.copy(x))
        
    return x, np.array(history)

# Example usage within a simulation loop:
# for t in range(NUM_TEST_SAMPLES):
#     local_opinions = initial_llrs[t, :]
#     final_consensus, conv_history = run_gossip_consensus(local_opinions, adj)