#!/bin/bash
#SBATCH --output=./results/log-files/all.log
#SBATCH -C EPYC-7543
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:0

source /home/pvannostrand/anaconda3/bin/activate facet

RUNPATH=/home/pvannostrand/FACET
cd $RUNPATH

module load gurobi/9.5.0

# Expr 1: FACET Index Evaluation
## 1a Vary Sigma, set T=100 MaxDepth=None
python main.py --expr sigma --ds cancer glass magic spambase vertebral --method FACETIndex --it 0 1 2 3 4 5 6 7 8 9 --values 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 --fmod softaxilf

## 1b Vary Nrects, Set T=100 MaxDepth=None
python main.py --expr nrects --ds cancer glass magic spambase vertebral --method FACETIndex --it 0 1 2 3 4 5 6 7 8 9 --values 100 1000 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 --fmod softaxilf

## 1c Enumeration, Set T=100, MaxDepth=None
python main.py --expr enum --ds cancer glass magic spambase vertebral --method FACETIndex --it 0 1 2 3 4 5 6 7 8 9

# Expr 2: Compare Methods, set T=10 MaxDepth=5
python main.py --expr compare --ds cancer glass magic spambase vertebral --method FACETIndex OCEAN AFT MACE RFOCSE --it 0 1 2 3 4 5 6 7 8 9 --fmod softaxilf

# Expr 3: Vary Ntrees, set MaxDepth=5
python main.py --expr compare --ds cancer glass magic spambase vertebral --method FACETIndex OCEAN --values 10 50 100 200 300 400 500 --it 0 1 2 3 4 5 6 7 8 9 --fmod softaxilf
