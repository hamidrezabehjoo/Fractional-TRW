#!/bin/bash
#SBATCH --output=a.out
#SBATCH --ntasks=10
#SBATCH --nodes=1             
#SBATCH --time=99:00:00   
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=5gb
#SBATCH --account=chertkov
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_chertkov



module load python/3.11/3.11.4
source  /home/u14/hbehjoo/env11/bin/activate

## install requied packages
cd ~/Fractional-TRW/mrftools/
python3 setup.py install


cd ~/Fractional-TRW/fig01
python3 planar_graph_3x3.py


