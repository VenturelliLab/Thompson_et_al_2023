# Integrating a tailored recurrent neural network with Bayesian experimental design to optimize microbial community functions

![Alt text](Fig1/figure_1_2.png?raw=true "Design-test-learn strategy")

# Installation steps:

The current implementation depends on JAX (https://github.com/google/jax) which requires either Linux or macOS (Windows users must use the Windows Subsystem for Linux)

I recommend first installing Anaconda Python (https://www.anaconda.com/products/distribution)

Once Python is installed, download this repository and run:

cd Thompson_et_al_2023

python setup.py install

Once installed, try running the Tutorial.ipynb Jupyter notebook or any other Jupyter notebook to reproduce the results from the manuscript.

If using Windows Subsystem for Linux, Jupyter notebooks can be converted to Python scripts (.py) using the command:

jupyter nbconvert --to script [Name of Jupyter Notebook].ipynb
