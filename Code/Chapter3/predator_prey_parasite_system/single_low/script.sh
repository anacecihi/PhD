#!/bin/bash
#$ -cwd                     		# Run the code from the current directory
#$ -j y                    			# Merge the standard output and standard error
#$ -l h_rt=120:00:00         		# Limit each task to 3 days
#$ -l h_vmem=5G            	    	# Request 3 GB RAM
#$ -t 1-100               		    # job array

# load modules
module load anaconda3

# Create a variable with the path to where the script is located
script_python=$(echo /data/scratch/btx499/invasion/)

printf "\nRunning...\n"

for rp in $(seq 0.5 0.1 1); do for rx in $(seq 0.5 0.1 1); do python $script_python/mono_single_low.py $rp $rx $SGE_TASK_ID; done; done

printf "\n"
echo Next the second script"!"
printf "\n"
wait

python $script_python/average_files_mono_single_low.py # run "final.py" which is inside "script_python"

printf "\n"
echo Job finished"!"
printf "\n"