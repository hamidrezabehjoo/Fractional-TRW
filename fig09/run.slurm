#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------

#SBATCH --job-name=Fractional-TRW
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1             
#SBATCH --mem-per-cpu=10gb                
#SBATCH --time=99:99:99   
#SBATCH --partition=standard
#SBATCH --account=chertkov
 
# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Load required modules/libraries if needed
module purge
module load anaconda/2022.05

conda init bash 
source ~/.bashrc
conda activate zahra

### change to your script’s directory
#cd ~/Fractional-TRW/mrftools
#python3 setup.py install
### Run your work
cd ~/Fractional-TRW/convergence
python3 G3.py
