#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=grad-project
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%j.out
### REQUIRED. Specify the PI group for this job
#SBATCH --account=chertkov
### Optional. Request email when job begins and ends
### SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
# SBATCH --mail-user=hbehjoo@arizona.edu
### REQUIRED. Set the partition for your job.
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=1
### REQUIRED. Set the number of nodes
#SBATCH --nodes=4
### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=20gb
###GPU
#SBATCH --gres=gpu:0
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=50:01:00
 
 
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
cd ~/Fractional-TRW
python3 07.py
