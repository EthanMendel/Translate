#! /bin/bash
### Author: Ethan Mendel
### Specify queue to run
#PBS -q mamba
### Set the job name
#PBS -N my_job
### Specify the # of cpus, gpus, and RAM memory for your job.
#PBS -l nodes=1:ppn=1:gpus=1,mem=128GB
### Specify maximum running time in hours:minutes:seconds
#PBS -l walltime=301:01:01
### pass the full environment of where sample.sh is
#PBS -V
### Create two log files .o that stores any outputs by the python file and .e that stores any errors
#PBS -o $PBS_JOBID.o 
#PBS -e $PBS_JOBID.e
#PBS -M emendel@uncc.edu
#
# ===== END PBS OPTIONS =====


### IMPORTANT: load Python 3 environment with Pytorch and cuda enabled
module load tensorflow/1.14-anaconda3-cuda10.0
echo "loaded module"
### Go to the directory of the sample.sh file
cd $PBS_O_WORKDIR
### Make a folder for job_logs if one doesn't exist
mkdir -p job_logs
### Run the python file
echo "running code"
python hebrew_to_english.py
echo "finished running"
### move the log files inside the folder
mv $PBS_JOBID.o job_logs/$PBS_JOBID.o
mv $PBS_JOBID.e job_logs/$PBS_JOBID.e
