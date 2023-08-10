# CCSD - Combinatorial Complex Score-based Diffusion Model using Stochastic Differential Equations

<p align="center"><img src="https://github.com/AdrienC21/CCSD/blob/main/logo.png?raw=true" alt="CCSD_logo" width="600"/></p>

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**CCSD** is a sophisticated score-based diffusion model designed to generate Combinatorial Complexes using Stochastic Differential Equations. This cutting-edge approach enables the generation of complex objects with higher-order structures and relations, thereby enhancing our ability to learn underlying distributions and produce more realistic objects.

## Table of Contents

- [CCSD - Combinatorial Complex Score-based Diffusion Model using Stochastic Differential Equations](#ccsd---combinatorial-complex-score-based-diffusion-model-using-stochastic-differential-equations)
  - [Table of Contents](#table-of-contents)
  - [CCSD](#ccsd)
    - [Introduction](#introduction)
    - [Why CCSD?](#why-ccsd)
    - [Author](#author)
  - [Installation](#installation)
    - [Using pip](#using-pip)
    - [Manually](#manually)
    - [Next steps](#next-steps)
  - [Dependencies](#dependencies)
  - [Testing](#testing)
  - [Usage](#usage)
    - [General](#general)
    - [Command line](#command-line)
    - [CCSD class](#ccsd-class)
  - [Commons errors](#commons-errors)
    - [Installation of MOSES](#installation-of-moses)
    - [Error due to MOSES](#error-due-to-moses)
    - [Error due to TopoNetX when running the tests](#error-due-to-toponetx-when-running-the-tests)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)

## CCSD

### Introduction

Complex object generation is a challenging problem with application in various fields such as drug discovery. The CCSD model offers a novel approach to tackle this problem by leveraging Diffusion Models and Stochastic Differential Equations to generate Combinatorial Complexes (CC). This topological structure generalizes the different mathematical stuctures used in Topological/Geometric Deep Learning to represent complex objects with higher-order structures and relations. The integration of the higher-order domain during the generation enhances the learning of the underlying distribution of the data and thus, allows for better data generation.

If you find this project interesting, we would appreciate your support by leaving a star ⭐ on this [GitHub repository](https://github.com/AdrienC21/CCSD).

Code still in **Alpha version!**

### Why CCSD?

CCSD stands out from traditional complex object generation models due to the following key advantages:

- *Combinatorial Complexes:* The model generates Combinatorial Complexes, enabling the synthesis of complex objects with rich structures and relationships.

- *Score-Based Diffusion:* CCSD utilizes score-based diffusion techniques, allowing for efficient, high-quality and state-of-the-art complex object generation.

- *Enhanced "Realism":* By incorporating higher-order structures, the generated objects are more representative of the underlying data distribution.

Also, this **repository is highly documented and commented**, which makes it easy to use, understand, deploy, and which offers endless possibilities for improvements.

### Author

The research has been conducted by **Adrien Carrel** as part of his requirements for the MSc degree in Advanced Computing of Imperial College London, United Kingdom, and his requirements for the MEng in Applied Mathematics (Diplôme d'Ingénieur) at CentraleSupélec, France.

## Installation

### Using pip

To get started with CCSD, you can install the package using pip by typing the command:

```bash
pip install ccsd
```

### Manually

If you encounter, if you want to use the latest version, or if you prefer the command line interface, you can use it locally by cloning or forking this repository to your local machine.

```bash
git clone https://github.com/AdrienC21/CCSD.git
```

### Next steps

Then, install the dependencies (see the section **Dependencies** below).

When installing PyTorch and its componenents, or when install TopoModelX along with TopoNetX, run the commands below:

```bash
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
```

where ${CUDA} could be `cu117`, `cu118`, or `cpu` if you want to use the CPU. For GPU, we recommend `cu118`. Also, TopoModelX should be installed after TopoNetX to avoid versioning issues.

To test your installation, refer to the section **Testing** below.

If you encounter an error, please refer to the section **Commons errors** below.

## Dependencies

CCSD requires a recent version of Python, probably 3.7, but preferably **3.10 or higher**.

It also requires the following dependencies:

- dill>=0.3.6
- easydict>=1.10
- freezegun>=1.2.2
- hypernetx>=1.2.5
- imageio>=2.31.1
- joblib>=1.3.1
- kaleido>=0.1.0.post1
- matplotlib>=3.7.2
- molsets>=0.3.1
- networkx>=2.8.8
- numpy>=1.24.4
- pandas>=2.0.3
- plotly>=5.15.0
- pyemd>=1.0.0
- pytest>=7.4.0
- pytz>=2023.3
- PyYAML>=6.0.1
- rdkit>=2023.3.2
- scikit_learn>=1.3.0
- scipy>=1.11.1
- TopoNetX>=0.0.2
- torch>=2.0.1
- tqdm>=4.65.0
- ~aleido>=0.2.1

Please make sure you have the required dependencies installed before using CCSD.

You can install all of them by running the command:

```bash
pip install -r requirements.txt
```

## Testing

To ensure the correctness and robustness of CCSD and to allow researchers to build upon this tool, we have provided an extensive test suite. To run the tests, execute the following command:

```bash
pytest tests/ -W ignore::DeprecationWarning
```

If you encounter an error, please refer to the section **Commons errors** below.

## Usage

### General

To use CCSD, follow the steps outlined below:

**Edit your general configurations:**

- Edit the file `config\general_config.py` to provide your *wandb* (Weights & Biases) information (if you want to use wandb), your timezone, and some other general parameters.

**Execute the code:**

- You can either use the command line or directly the CCSD class (more information in the subsections below).

**Combinatorial Complexes:**

To generate combinatorial complexes and not graphs, just put the parameter `is_cc` to True in your configuration files, specify the `d_min` and `d_max` parameters of your dataset (see the thesis for more information on that), and pick a ScoreNetwork model for the rank2 score predictions (see example of configurations).

**Figures and output:**

The figures, the graphs, combinatorial complexes, molecules, etc, will all be saved into a `samples` folder, and in the logs folders `logs_sample` and `logs_train`.

The output in the command line should look like below, where a logo is printed, the current experiment information, and then the training/sampling information:

```bash
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP5JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJYPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP?                                7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPP5YY5PPPPPPPPPP5YY5PPPPPPPPPPPPPPPPPP?           .^!7?????7!^.        7PPPPPPPPPPPPPP
PPPPPPPPPPPPPP5^.  ~JYYY55557^....^?PPPPPPPPPPPPPPPP?        :!J5PPPPPPPPPPP5J!.     7PPPPPPPPPPPPPP
PPPPPPPPPPPPPP?   :^      ~^        ~PPPPPPPPPPPPPPP?      .?5PPPPPPPPPPPPPPPPP5!    7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPP?^:^?J??777?..      .:5PPPPPPPPPPPPPP?     ^5PPPPPP5?~^^~!?5PPPY!:    7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPG5^:^.   ^: :75PPPPPPPPPPPP?    :5PPPPPP7.       .7!:       7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPP5!.  !?^^~JJ~ .^!^^^~75PPPPP?    7PPPPPPJ                    7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPY!.  ~YPPPPPPPPY^       .?PPPP?    JPPPPPP!                    7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPP5557.  ~YPPPPPPPPPPJ         :PPPP?    !PPPPPPJ                    7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPJ~....:~YPPPPPPPPPPPP5:        7PPPP?    .YPPPPPP?.       :?7^.      7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPJ       :5PPPPPPPPPPPPP57^...:~JPPPPP?     .JPPPPPP5?!~~~!?5PPP5J!.   7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPJ       .5PPPPPPPPPPPPPPPP555PPPPPPPP?       !5PPPPPPPPPPPPPPPPP57    7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPJ^...:~YPPPPPPPPPPPPPPPPPPPPPPPPPPPP?        .~?5PPPPPPPPPPP5J~.     7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPP5555PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP?           .:~!7??77!~:.        7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP?                                7PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP5YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY5PPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP   _____ _____  _____ _____   PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  / ____/ ____|/ ____|  __ \  PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP | |   | |    | (___ | |  | | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP | |   | |     \___ \| |  | | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP | |___| |____ ____) | |__| | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  \_____\_____|_____/|_____/  PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP                              PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
----------------------------------------------------------------------------------------------------
Current experiment:

        type: train
        config: qm9_test
        comment: 
        seed: 42

...
```

### Command line

**Help:**

- Get more information by typing:

```bash
python main.py --help
```

**Train a model:**

- Define your own config or use one from the **config** folder.

- Use the command:

```bash
python main.py --type train --config <config_name>
```

where `<config_name>` should be the name of your configuration file.

- If you want to sample directly after the training of your model, just provide in the configuration file the sampling parameters.

**Sample from a model:**

- Define your own sample config or use one from the **config** folder.

- Use the command:

```bash
python main.py --type sample --config <config_name>
```

where `<config_name>` should be the name of your sampling configuration file.

**Other:**

- Other parameters include:

`--comment COMMENT`: A single line comment for the experiment

`--seed SEED`: Random seed for reproducibility

`--folder FOLDER`: Directory to save the results, load checkpoints, load config, etc

- To provide additional information about the GPU, you can provide the ID(s) of the GPU(s) to use by typing:

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --type <type> --config <config_name>
```

if you want to use the GPUs 0 and 1, or to only use the GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --type <type> --config <config_name>
```

### CCSD class

- Run an experiment by adapting the code below:

```python
from ccsd import CCSD


params = {
  "type": "train",
  "config": "qm9_cc",
  "folder": "./",  # optional
  "comment": "test experiment",  # optional
  "seed": 42  # optional
}
diffusion_model = CCSD(**params)  # define the object
diffusion_model.run()  # run the experiment
```

- To get more information about the parameters and the other functions availables, you can ask for help by typing:

```python
from ccsd import CCSD

help(CCSD)
```

## Commons errors

### Installation of MOSES

If you encounter an error during the installation of MOSES, please follow the steps below:

First, if you are on Windows, make sure you have installed the *C++ Dev kit via Visual Studio 2022 community*.

Then, for all users, install rdkit, Cython, and pomegranate using the commands:

```bash
pip install rdkit
pip install Cython
pip install pomegranate
```

Finally, either install MOSES directly using:

```bash
pip install molsets
```

**Or** install it manually if it doesn't work by typing the following commands:

```bash
git lfs install
git clone https://github.com/molecularsets/moses.git
cd moses
python setup.py install
```

### Error due to MOSES

If you get an error related to a `._append` method that no longer exists in Pandas and that is still used in the MOSES package, please replace in the MOSES package the file `utils.py` with the one provided in this repository: **fixes\utils.py**. The MOSES `utils.py` file should be located somewhere like:

`C:\Users\<username>\miniconda3\lib\site-packages\moses\metrics\utils.py` or `miniconda\lib\python3.11\site-packages\molsets-1.0-py3.11.egg\moses\metrics\utils.py`

### Error due to TopoNetX when running the tests

Replace the file `combinatorial_complex.py` of TopoNetX by the updated one provided in this repository: **fixes\combinatorial_complex.py**

The file in TopoNetX should be located somewhere like:

`C:\Users\<username>\miniconda3\Lib\site-packages\toponetx\classes\combinatorial_complex.py` or `miniconda\lib\python3.11\site-packages\toponetx\classes\combinatorial_complex.py`

## Citation

If you use CCSD in your research or work, please consider citing it using the following BibTeX entry:

```bibtex
Carrel, A. (2023). CCSD - Combinatorial Complex Score-based Diffusion model using stochastic differential equations. (Version 1.0.0) [Computer software]. https://github.com/AdrienC21/CCSD
```

## Acknowledgement

The Laboratory for Computational Physiology (LCP) at the Massachusetts Institute of Technology (MIT) for hosting me during the redaction of my thesis.

Dr. Tolga Birdal for his supervision and for the valuable advice and ressources that he provided.

All my friends and my family for the support.

Logo created by me using the icon: "topology" icon by VectorsLab from Noun Project CC BY 3.0.

## License

CCSD is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). Feel free to use and modify the code as per the terms of the license.
