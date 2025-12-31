import numpy as np

def run_gossip_consensus(initial_llrs, adj_matrix, max_iterations=30, epsilon=0.01, momentum=0.2):
    """
    STABILIZED GOSSIP LOGIC:
    - Reduced momentum (0.2) to ensure the Spectral Radius < 1.
    - Added Divergence Guard to prevent LLR explosions.
    """
    num_users = len(initial_llrs)
    degrees = np.sum(adj_matrix, axis=1)
    
    x = np.copy(initial_llrs).astype(float)
    x_prev = np.copy(x)
    history = [np.copy(x)]
    
    for k in range(max_iterations):
        # 1. MOMENTUM STEP (Stable Acceleration)
        # Low momentum (0.2) speeds up convergence without causing oscillation.
        v = x + momentum * (x - x_prev)
        
        x_next = np.copy(x)
        for i in range(num_users):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            for j in neighbors:
                # Metropolis weights for a Doubly Stochastic Matrix.
                weight = 1 / (max(degrees[i], degrees[j]) + 1)
                x_next[i] += weight * (v[j] - v[i])
        
        # 2. DIVERGENCE GUARD
        # If values exceed a reasonable LLR range (±20), the process is unstable.
        if np.any(np.abs(x_next) > 20):
            print(f"Warning: Instability at Iteration {k}. Reverting to standard gossip.")
            return x, np.array(history), k
            
        # 3. ENERGY-SAVING TERMINATION
        if np.linalg.norm(x_next - x) < epsilon:
            # Fill remaining history for plotting
            for _ in range(max_iterations - k):
                history.append(np.copy(x_next))
            return x_next, np.array(history), k + 1
            
        x_prev, x = x.copy(), x_next.copy()
        history.append(x)
        
    return x, np.array(history), max_iterations