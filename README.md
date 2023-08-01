# CCSD - Combinatorial Complex Score-based Diffusion Model using Stochastic Differential Equations

<p align="center"><img src="https://github.com/AdrienC21/CCSD/blob/main/logo.png?raw=true" alt="CCSD_logo" width="600"/></p>

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**CCSD** is a sophisticated score-based diffusion model designed to generate Combinatorial Complexes using Stochastic Differential Equations. This cutting-edge approach enables the generation of complex objects with higher-order structures and relations, thereby enhancing our ability to learn underlying distributions and produce more realistic objects.

## Table of Contents

- [CCSD - Combinatorial Complex Score-based Diffusion Model using Stochastic Differential Equations](#ccsd---combinatorial-complex-score-based-diffusion-model-using-stochastic-differential-equations)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Documentation](#documentation)
  - [Usage](#usage)
  - [Why CCSD?](#why-ccsd)
  - [Testing](#testing)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)

## Introduction

Complex object generation is a challenging problem with application in various fields such as drug discovery. The CCSD model offers a novel approach to tackle this problem by leveraging Diffusion Models and Stochastic Differential Equations to generate Combinatorial Complexes (CC). This topological structure generalizes the different mathematical stuctures used in Topological/Geometric Deep Learning to represent complex objects with higher-order structures and relations. The integration of the higher-order domain during the generation enhances the learning of the underlying distribution of the data and thus, allows for better data generation.

If you find this project interesting, we would appreciate your support by leaving a star ⭐ on this [GitHub repository](https://github.com/AdrienC21/CCSD).

Code still in **Alpha version!**

**Author:** Adrien Carrel

## Installation

To get started with CCSD, you can clone or fork this repository to your local machine.

```bash
git clone https://github.com/AdrienC21/CCSD.git
```

- [To be completed]

## Dependencies

CCSD requires the following dependencies:

- [To be completed]

Please make sure you have the required dependencies installed before using CCSD.

You can install all of them by running the command:


pip install -r requirements.txt
```

## Documentation

For detailed information on using CCSD, refer to the documentation:

- [To be completed]

## Usage

To use CCSD, follow the steps outlined below:

- [To be completed]

1. [Step 1]

2. [Step 2]

3. [Step 3]

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

## Why CCSD?

CCSD stands out from traditional complex object generation models due to the following key advantages:

- *Combinatorial Complexes:* The model generates Combinatorial Complexes, enabling the synthesis of complex objects with rich structures and relationships.

- *Score-Based Diffusion:* CCSD utilizes score-based diffusion techniques, allowing for efficient, high-quality and state-of-the-art complex object generation.

- *Enhanced "Realism":* By incorporating higher-order structures, the generated objects are more representative of the underlying data distribution.

- [To be completed]

## Testing

To ensure the correctness and robustness of CCSD and to allow researchers to build upon this tool, we have provided an extensive test suite. To run the tests, execute the following command:

```bash
pytest tests/ -W ignore::DeprecationWarning
```

## Citation

If you use CCSD in your research or work, please consider citing it using the following BibTeX entry:

```bibtex
[To be completed, and complete the citation information in the CITATION.cff file provided in the repository.]
```

## Acknowledgement

Logo created by me using the icon: "topology" icon by VectorsLab from Noun Project CC BY 3.0.

## License

CCSD is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). Feel free to use and modify the code as per the terms of the license.
