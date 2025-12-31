# dist_phase2_hmm.py
import numpy as np
from hmmlearn import hmm

def train_node_hmms(rx_energies):
    """Each node trains its own HMM brain independently."""
    num_users = rx_energies.shape[1]
    node_models = []
    
    print(f"--- Training {num_users} Independent Local HMMs ---")
    for u in range(num_users):
        # GaussianHMM models the continuous energy values
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
        user_data = rx_energies[:, u].reshape(-1, 1)
        model.fit(user_data)
        node_models.append(model)
    
    return node_models

def get_initial_llrs(models, test_energies):
    """Nodes calculate their initial 'Soft Opinion' (LLR) before talking."""
    num_samples = test_energies.shape[0]
    num_users = test_energies.shape[1]
    initial_llrs = np.zeros((num_samples, num_users))
    
    for u in range(num_users):
        model = models[u]
        data = test_energies[:, u].reshape(-1, 1)
        
        # Identify which state has higher mean (The 'ON' state)
        idx_on = 1 if model.means_[1][0] > model.means_[0][0] else 0
        idx_off = 1 - idx_on
            
        # Predict probability of each state for every sample
        probs = model.predict_proba(data)
        
        # Calculate LLR = log(P_on / P_off)
        p_on = probs[:, idx_on] + 1e-10
        p_off = probs[:, idx_off] + 1e-10
        initial_llrs[:, u] = np.log(p_on / p_off)
        
    return initial_llrs