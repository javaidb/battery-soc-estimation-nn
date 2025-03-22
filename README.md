# Battery State Estimation using Neural Nets (SOC)

> This repository is a part of a series of repositories aimed at deepening personal understanding of lithium-ion battery management systems along with practical implementations and contexts. Through this repo, I explore advanced battery State of Charge (SOC) estimation using deep learning techniques, mainly for me to expand on experience learnt in career + courses + self-learning while identifying areas for self-improvement in my own knowledge and skills. It is designed more so as a sandbox for me to develop, test and implement state estimation techniques for various sample li-ion batteries. This project implements an LSTM-based architecture with attention mechanisms for accurate real-time SOC prediction.

## Project Overview

This project implements a neural network-based approach to battery State of Charge estimation, using the following components:
- Bidirectional LSTM for temporal pattern recognition (i.e. timeseries format with various features throughout)
- Multi-head attention mechanisms for feature importance weighting
- Physics-informed architecture incorporating battery behavior (OCV, overpotential, etc.)
- MLflow integration for experiment tracking

### Key Features

- **Advanced Architecture**:
  - Bidirectional LSTM layers for temporal dependencies
  - Multi-head attention mechanism
  - Skip connections for improved gradient flow
  - Layer normalization for training stability

- **Battery-Specific Design**:
  - Overpotential prediction
  - Open Circuit Voltage (OCV) integration
  - Temperature compensation
  - Current-based dynamics modeling

- **Development Features**:
  - GPU acceleration support
  - MLflow experiment tracking
  - Hyperparameter optimization
  - Docker containerization

## Getting Started

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU (optional, but recommended)
- NVIDIA Container Toolkit (for GPU support)
- Git (for version control)

### Project Structure
```
├── data/                      # Data directory
│   ├── Mendeley/             # Battery dataset
│   ├── processed/            # Processed data
│   └── README.md             # Data format documentation
├── model_training/           # Training infrastructure
│   ├── src/                 # Source code
│   ├── config.yaml          # Training configuration
│   └── docker-compose.yml   # Container configuration
├── preprocessing/           # Data preprocessing scripts
└── notebooks/              # Jupyter notebooks for analysis
```

### Workflow

1. **Setup Environment**:
   ```bash
   # Clone repository
   git clone https://github.com/javaidb/battery-soc-estimation-nn.git
   cd battery-soc-estimation-nn
   
   # Build training image
   cd model_training
   docker compose build cuda_model_trainer
   ```

2. **Data Preparation**:
   ```bash
   # Start preprocessing service
   docker compose up preprocessor
   
   # Data will be processed from data/Mendeley/1_raw
   # through to data/Mendeley/3_processed
   ```

3. **Configure Training** (`model_training/config.yaml`):
    Edit `model_training/config.yaml` to modify:
    - Training parameters
    - Model architecture
    - Hyperparameter ranges
    - Data paths

   ```yaml
   experiment: overpotential_model_test
   num_trials: 30
   max_epochs: 50
   max_time: 6
   
   model:
     input_size: 3
     hidden_size:
       hyperparameter_type: integer
       name: hidden_size
       low: 3
       high: 30
     output_size: 1
   ```

4. **Start Training Pipeline**:
   ```bash
   # Launch MLflow server
   docker compose up -d mlflow
   
   # Start training (with GPU support)
   docker compose up cuda_model_trainer
   
   # Monitor training progress at http://localhost:5000
   ```

5. **Monitor & Manage**:
   ```bash
   # View training logs
   docker compose logs -f cuda_model_trainer
   
   # Check GPU usage
   nvidia-smi -l 1
   
   # Stop all services
   docker compose down
   ```

### Using Trained Models

```python
from src.cell_model import CellModel

# Load model
model = CellModel(
    input_size=3,
    hidden_size=64,
    output_size=1,
    num_lstm_layers=2
)
model.load_state_dict(torch.load("training/output/best_model.pth"))

# Make predictions (input shape: [batch_size, seq_length, features])
predictions = model(input_data)
```

## Dataset

Uses the LG 18650HG2 Li-ion battery dataset from Mendeley:
- 3Ah nominal capacity
- Various discharge rates
- Temperature measurements
- Voltage and current profiles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If using the dataset, please cite:
```
Kollmeyer, Philip; Vidal, Carlos; Naguib, Mina; Skells, Michael (2020), 
"LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script", 
Mendeley Data, V3, doi: 10.17632/cp3473x7xv.3
```

## Acknowledgments

- Dataset providers from Mendeley
- PyTorch and MLflow communities
- Battery research community, incl. @xiansee for reference
