#!/bin/bash
#$ -cwd                     		# Run the code from the current directory
#$ -j y                    			# Merge the standard output and standard error
#$ -l h_rt=240:00:00         		# Limit each task to 3 days
#$ -l h_vmem=3G            			# Request 3 GB RAM
#$ -t 1-100               		    # job array

# load modules
module load anaconda3

# Create a variable with the path to where the script is located
script_python=$(echo /data/scratch/btx499/stochastic)

printf "\nRunning...\n"

for Qx in $(seq 0 0.01 1); do for Qy in $(seq 0 0.01 1); do python $script_python/stochastic.py $Qx $Qy $SGE_TASK_ID; done; done

printf "\n"
echo Next the second script"!"
printf "\n"
wait

python $script_python/average_timepoints.py # run "final.py" which is inside "script_python"