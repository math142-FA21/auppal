# Math 142 Repo
This is the Main Repo for Abhi Uppal's work for Math 142, Differential Geometry at Harvey Mudd College (Fall 2021). This includes homework, reading repsonses, and code for the midterm and final projects.

This README will primarily be dedicated to the setup for the project

# Prerequisites

The necessary software needed to build the project will be listed here and will be expanded if necessary. As of right now, it is just

- [git](https://git-scm.com/)
- Late model [Anaconda](https://www.anaconda.com/products/individual)

## Conda Environment

It is advisable to install this project in a clean conda environment. This project is installed as an editable module, so anything in the `thesis` folder can be imported just as a regular Pyhton package. Dependencies and package information are listed in `setup.py`. In order to build the environment, run the following commands in the root directory of this repository:

```bash
conda create -n diffgeo python=3.8

conda activate diffgeo

pip install -e .
```

Once this is done, you can deactivate the environment with `conda deactivate` and reactivate it with `conda activate diffgeo`.
