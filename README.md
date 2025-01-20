# Battery State Estimation using Neural Nets (SOC)

> This repository is a part of a series of repositories aimed at deepening personal understanding of lithium-ion battery management systems along with practical implementations and contexts. Through this repo, SOC estimation techniques using NNs are explored for me to expand on experience learnt in career + courses + self-learning while identifying areas for self-improvement in my own knowledge and skills. It is designed more so as a sandbox for me to develop, test and implement state estimation techniques for various sample li-ion batteries.

## Getting Started

### Prerequisites

To run this project, you will need Python installed on your machine along with the required libraries. Current version utilizations being utilzied locally are as follows:
- Python (3.11.7)
- Virtual environment created using Python 3.11 (Not required but suggested)

To create and activate virtual environment:

```bash
python -m venv .venv_soc_nn
source .venv_soc_nn/bin/activate  # Linux/Mac
# or
.venv_soc_nn\Scripts\activate  # Windows
python -m ipykernel install --name ".venv_soc_nn" --display-name "NN (SOC est.)" --user
```
Otherwise, you can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```
