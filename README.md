### Abstract
We investigate the emergence of binary encoding within the latent space of deep-neural-network classifiers.
Such binary encoding is induced by the integration of a linear penultimate layer, which employs during training a loss function specifically designed to compress the latent representations. 
As a result of a trade-off between compression and information retention, the network learns to assume only one of two possible values for each dimension in the latent space.
The binary encoding is provoked by the collapse of all representations of the same class to the same point, which corresponds to the vertex of a hypercube, thereby creating the encoding.
We demonstrate that the emergence of binary encoding significantly enhances robustness, reliability and accuracy of the network.

### Dependencies 
Code was tested on Python 3.11.8, PyTorch 2.2.1, NumPy 1.26, scikit-learn 1.2.2, and SciPy 1.11.4.

### Install
```
pip install .
```

### Reproduce results 
Results can be reproduced on a GPU cluster using a slurm script that we provide in `scripts/run_slurm_jobs.sh`. The script is assumed to be run in a conda environment named _bin_enc_ where the dependencies indicated above and the package defined in this repository is installed. We utilized NVIDIA A100 GPUs, as indicated in the _gres_ argument in the script. Training for the CIFAR10 and CIFAR100 datasets can be run with the commands:

```
scripts/run_slurm_jobs.sh configs/cifar10.yml cifar10 datasets results jobs_outputs
scripts/run_slurm_jobs.sh configs/cifar100.yml cifar100 datasets results jobs_outputs
```

where the `cifar10.yml` and `cifar100.yml` files contain all training hyperparameters, the 'datasets' directory is created to store the loaded datasets, the `results` directory is created to store all training results, and the `jobs_output` directory is created to store all slurms jobs outputs. 
For each of the 5 experiments, a number of training results are produced with different learning rates, and the best results in each experiment can be picked with `scripts/find_best_results.py`:

```
python scripts/find_best_results.py --results-dir results/cifar10 --output-dir results/cifar10
python scripts/find_best_results.py --results-dir results/cifar100 --output-dir results/cifar100
```

Results can finally be visualized using the Jupyter notebook `notebooks/plots.ipynb`.

### Demonstrative training
A short training to demonstate the emergence of binary encoding can be done with the _emergence_binary_encoding_ package in the Jupyter notebook `notebooks/train.ipynb`.

