# D4: Distance Diffusion for a Truly Equivariant Molecular Design

## Requirements

The code was tested with Python 3.11.7 and with CUDA 11.4.0. The requirements can be installed by executing 
- Download anaconda/miniconda if needed
- Create a new environment through the given environment files with the following command:
    ```bash
    conda env create -f <env_file>.yml
    ```
    where \<env_file\> is the name of the environment file to use. It is possible to install dependencies for CPU with `environment_cpu.yml` or for GPU with `environment_cuda.yml`.
- Install this package with the following command:
    ```bash
    pip install -e .
    ```
    which will compile required cython and c++ code.

### Running experiments

The experiments can be run with different modality, the basic command is:
```bash
    python main.py +preset=<preset> 
```
where the value preset could be chosen between qm9_distance and gdb13_distance.
After the \<preset>\ it's possible to mention the \<seed>\, the modality of running \<mode>\ between \<eval>\, \<train+eval>\ and \<train>\. 
All the parameters relative to the training and evaluation phase could be seen inside ./config\ folder.

### Datasets

QM9 will be downloaded automatically to a new directory ./datasets when running an experiment. 
Since for GDB13 only a subset of the entire dataset is used, for reproducibility, this could be found directly inside ./dataset/GDB13/raw/ folder.

### Checkpoints

Checkpoints are saved in a new directory ./checkpoints. The code is provided with best checkpoints used for experiments in both QM9 and GDB13 case. The file associated to the best checkpoints is called "best_*.ckpt". 
