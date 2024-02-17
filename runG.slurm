#!/bin/bash
#SBATCH --output= planar_graph_3x3.out
#SBATCH --ntasks=10
#SBATCH --nodes=1             
#SBATCH --time=99:99:99   
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=5gb
#SBATCH --partition=standard
#SBATCH --account=chertkov



module load python/3.11/3.11.4
source  /home/u14/hbehjoo/env11/bin/activate
cd ~/mrftools
python3 setup.py install

cd ~/Fractional-TRW/fig01
python3 planar_graph_3x3.py

