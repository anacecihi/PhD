#!/bin/bash
#$ -cwd                     		# Run the code from the current directory
#$ -j y                    			# Merge the standard output and standard error
#$ -l h_rt=240:00:00         		# Limit each task to 3 days
#$ -l h_vmem=10G            		# Request 3 GB RAM
#$ -t 1-100               		    # job array

# load modules
module load anaconda3

# Create a variable with the path to where the script is located
script_python=$(echo /data/scratch/btx499/active)

printf "\nRunning...\n"

for rp1 in $(seq 0 0.1 1); do for rp2 in $(seq 0 0.1 1); do for rx in $(seq 0 0.1 1); do python $script_python/active.py $rp1 $rp2 $rx $SGE_TASK_ID; done; done; done

printf "\n"
echo Next the second script"!"
printf "\n"
wait

python $script_python/average_files_active.py # run "final.py" which is inside "script_python"

printf "\n"
echo Job finished"!"
printf "\n"