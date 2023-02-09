#!/bin/bash
#$ -cwd                     		# Run the code from the current directory
#$ -j y                    			# Merge the standard output and standard error
#$ -l h_rt=120:00:00         		# Limit each task to 3 days
#$ -l h_vmem=10G            		# Request 3 GB RAM
#$ -t 1-100                		    # job array (e.g. 5 repeats)

##############################################################################
## 										   				                    ##
## Contact Ana Cecilia Hijar Islas for any doubts about this script:        ##
## a.c.hijarislas@qmul.ac.uk			                                    ##
##                                                                          ##
##############################################################################

# load modules
module load anaconda3

# Create a variable with the path to where the script is located
script_python=$(echo /data/scratch/btx499/eco_evo)

echo The analyses will be carried out in the directory $script_python
printf "\n"

#########
## Run ##
#########
printf "\nRunning...\n"

python $script_python/eco_evo.py $SGE_TASK_ID # run "test_apocrita" which is inside "script_python"

printf "\n"
echo Job finished"!"
printf "\n"