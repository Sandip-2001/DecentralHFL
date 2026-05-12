# DecentralHFL: Zero-Trust Heterogeneous Federated Learning

This repository introduces **DecentralHFL**, a Zero-Trust Web3 architecture that completely replaces the centralized aggregation server in Heterogeneous Federated Learning (HFL) with an immutable Motoko smart contract on the Internet Computer (IC) blockchain. 

*Note: This codebase is a major architectural fork and security enhancement of the original [AugHFL framework](https://github.com/FangXiuwen/AugHFL) (ICCV 2023).*

## ⚠️ The Problem: Centralized Vulnerabilities
Current state-of-the-art HFL frameworks rely on a fragile "Honest Server" assumption to compute Kullback-Leibler (KL) divergence trust weights. During our analysis, we uncovered a **Fatal Centralized Trade-Off**:
1. **NaN Gradient Explosion:** If the central server lacks native input sanitization, a 50% Byzantine sign-flip attack forces negative probabilities into the aggregation, causing a catastrophic math failure that instantly destroys the neural weights of all clients (crashing accuracy to 0.00%).
2. **Vanishing Gradients:** If standard tensor clamps are used to prevent the explosion, the distributions flatten, halting federated communication entirely. 
3. **Server Hacks:** Centralized Python scripts allow attackers to trivially bypass trust calculations by altering the server-side aggregation weights.

## 🛡️ The Solution: DecentralHFL
DecentralHFL solves this paradox by migrating the consensus mechanism natively on-chain. 
* **Edge Nodes (Local Training):** PyTorch models (ResNet10, ResNet12, ShuffleNet, MobileNetV2) train locally on private data.
* **The Bridge:** Raw multi-dimensional tensors are flattened and transmitted via `ic.candid` binary serialization.
* **Blockchain Consensus (Collaborative Update):** A WebAssembly (Wasm) smart contract natively detects anomalies, slashes negative logits, and executes objective KL trust-scoring on-chain, rendering trust-score manipulation cryptographically impossible.

## 📊 Datasets & Hyperparameters
Experiments are engineered with a specific 1:2 Public-to-Private Ratio to test absolute system robustness:
* **Private Data:** CIFAR-10-C (2,000 images per client)
* **Public Data:** CIFAR-100 (1,000 images for distillation)

## 🚀 Quick Start (Split-Architecture Execution)

### Prerequisites
Before running the project, ensure you have Anaconda/Miniconda installed for the Python environment, and the Internet Computer SDK (`dfx`) installed for the blockchain environment.

```bash
# Install DFX (macOS/Linux)
sh -ci "$(curl -fsSL https://internetcomputer.org/install.sh)"
```

Because this is a Web3 pipeline, you must run the Blockchain server and the PyTorch clients simultaneously in two separate terminals.

**Terminal 1: Start the Web3 Blockchain (Internet Computer)**
```bash
# Navigate to the smart contract directory
cd AugHFL/fl_server
# Start the local replica and deploy the Motoko contract
dfx start --background --clean
dfx deploy
```
**Terminal 2: Run the PyTorch Federated Loop**
```
# Navigate to the main project directory
cd AugHFL

# Activate your conda environment 
conda activate aughfl_env

# Install the required Python dependencies
pip install -r requirements.txt

# Generate the corrupted datasets
cd Dataset
python3 make_cifar_c.py

# Pretrain the local baseline models
cd ../Network
python3 pretrain.py

# Execute the DecentralHFL Workflow
cd ../HHF
python3 AugHFL.py
```
### 📈 Comparison & Threat Scenarios:
The AugHFL.py execution script can be configured to run the following threat models:
DecentralHFL (Web3 Defense): The IC smart contract quarantines a 50% Byzantine sign-flip attack. Honest networks survive at ~48.27% accuracy.
Central Server Hack: A compromised Python server forces poisoned logits into the network. The system suffers a fatal NaN explosion (0.00% accuracy).
Clean Baselines: Execution without active attackers to validate mathematical fidelity.
### 📝 Citation
If you use the baseline data-corruption or augmentation methodologies, please credit the original AugHFL authors:
@inproceedings{fang2023robust,
  title={Robust heterogeneous federated learning under data corruption},
  author={Fang, Xiuwen and Ye, Mang and Yang, Xiyuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5020--5030},
  year={2023}
}

### Why this README is perfect for your project:
1. **It highlights the exact bugs you found:** It explicitly mentions the *NaN Gradient Explosion* and the *Vanishing Gradients* directly in the introduction so visitors know exactly why your code is valuable.
2. **It explains the Split-Architecture:** Anyone who clones your code needs to know they have to run `dfx start` in one terminal and `python3 AugHFL.py` in another. This README provides those exact instructions.
3. **It respects the original authors:** By keeping their citation and clearly stating this is a "major architectural fork" of their ICCV 2023 paper, you maintain perfect academic integrity.
