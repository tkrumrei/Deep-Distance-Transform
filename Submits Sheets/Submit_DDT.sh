#!/bin/bash

#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node
#SBATCH --cpus-per-task=36          # the number cpus per task
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=24:00:00             # the max wallclock time (time limit your job will run)
 
#SBATCH --job-name=DDT_Train         # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=tkrumrei@uni-muenster.de # your mail address
 
# LOAD MODULES HERE IF REQUIRED
# imports needed: 
module purge

module load palma/2022a
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load GCCcore/11.3.0
module load numba/0.56.4
module load SciPy-bundle/2022.05
module load Pillow/9.1.1
module load methplotlib/0.20.1
module load tqdm/4.64.0
module load OpenCV/4.6.0-contrib
module load scikit-image/0.19.3

pip install --user torch torchvision
pip install --user PyQt5
pip install --user pyqtgraph
pip install --user fastremap
pip install --user roifile

# START THE APPLICATION
python  /home/t/tkrumrei/Deep-Dist-Transform/Deep_Dist_Transform_Train.py