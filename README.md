# **Accelerated-Consensus: A Stability-Guarded Distributed Sensing Framework**

**Project Title:** Application of Adaptive Probabilistic Models for Real-Time Distributed Spectrum Sensing in Cognitive Radio Networks

**Category:** Random Processes / Stochastic Systems Mini-Project (Python-based Simulation)

## **📌 Project Overview**

This project addresses a critical real-time challenge in 5G and Cognitive Radio: **Dynamic Spectrum Access**. In decentralized networks, multiple sensor nodes must decide if a primary frequency is occupied without a central hub.

We utilize a hierarchy of probabilistic models to achieve a decentralized agreement (consensus) that is more robust than any single-node observation, specifically addressing the trade-offs between accuracy, latency, and energy consumption.

## **🧠 Probabilistic Models Applied**

This software is built on three core stochastic foundations:

1. **Hidden Markov Models (HMM):** To model the temporal state transitions (Busy/Idle) of the Primary User.  
2. **Bayesian Inference (LLR):** Converting local energy detections into Log-Likelihood Ratios to represent probabilistic "belief."  
3. **Discrete-Time Markov Chains (DTMC):** Modeling the network-wide gossip iterations as a state-space evolution governed by a **Doubly Stochastic Matrix**.

## **🚀 True Criticism & Technical Optimizations**

Standard gossip algorithms often fail in hardware due to high latency and power drain. This project implements specific mathematical fixes to these real-world engineering failures:

* **Latency Fix (Momentum):** Standard gossip convergence is slow. We implemented a **second-order Markov process** using a "Heavy-Ball" momentum factor ($\\beta=0.2$) to accelerate information diffusion across the network.  
* **Battery Drain Fix (Epsilon-Stop):** Continuous communication is energy-expensive. We implemented an $\\epsilon$**\-termination criterion** that detects "Consensus in Probability," resulting in a **36.44% reduction in energy consumption** compared to fixed-iteration models.  
* **Stability Guard:** To prevent the wild numerical oscillations common in high-momentum feedback systems, we integrated a damping factor and divergence check to ensure the **Spectral Radius** remains within the unit circle.  
* **Hysteresis Decision Rule:** A $\\pm 0.5$ LLR buffer was added to navigate the "Uncertainty Zone," effectively reducing False Alarms in high-noise environments.

## **📚 Mathematical Concepts Explored**

* **Metropolis-Hastings Weights** for Graph Topology.  
* **Perron-Frobenius Theorem** for Matrix Convergence.  
* **Neyman-Pearson Criterion** for Detection vs. False Alarm.  
* **Rayleigh Fading & AWGN** for Channel Modeling.

## **🛠️ Software Requirements**

* **Language:** Python 3.8+  
* **Libraries:** NumPy (Matrix Dynamics), Matplotlib (Stochastic Visualization), Scikit-learn (Metrics).

## **🖥️ How to Run**

\# Clone the repository  
git clone https://github.com/IGSbv/distrubuted-consensus-based-spectrum-sensing.git

\# Run the Phase 4 Evaluation  
python dist\_phase4\_evaluation.py  
