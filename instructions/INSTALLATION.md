# Installation Instructions

Follow the directions below to install the core FACET system (published at SIGMOD'24) and optionally our WebUI (published at VLDB'24)

This repository is validated on Python 3.10, [requirements.txt](/requirements.txt) contains a list of required packages formatted for use with Anaconda

## Core System (Required)

- Download and install [Anaconda](https://www.anaconda.com/download/success)
- To run experiments with OCEAN, a state-of-the-art method we compare to, you will need a license for the Gurobi optimizer
  - Free academic licenses are available via [Gurobi's website](https://www.gurobi.com/academia/academic-program-and-licenses/)

```bash
# create a new anaconda environment
conda config --append channels conda-forge
conda config --append channels https://conda.anaconda.org/gurobi
conda create --name facet --file requirements.txt
conda activate facet
# if running SOTA comparison method MACE, install required solver
pysmt-install --z3 --confirm-agreement
# if running SOTA comparison method OCEAN install and activate the gurobi optimizer
pip install --ignore-installed gurobipy
grbgetkey <your_acadmic_license_key>
# test your installation
python main.py --expr simple
```

You're now ready to run experiments with FACET or continue to install our web app!

## Demo UI (Optional)

- FACET's web user interface is built using JavaScript and uses the Node Package Manager to install and build all dependencies
- To run the UI demo (published in VLDB 2024) download and install [Node.js](https://nodejs.org/en/download/prebuilt-installer)
  - Be sure to check `Automatically install the necessary tools` during Node installation

```bash
# install JavaScript packages for the UI
cd webapp
npm install
# lanch the webapp
npm run dev
```

Navigate to [http://localhost:5175/](http://localhost:5175) in your browser and checkout the UI!
