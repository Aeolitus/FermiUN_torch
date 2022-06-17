# SGE COMMANDS

# Determine the queues you want your job to run on.
# GPUs are available in graphix.q and graphit.q only!
#$ -q graphix.q

# Determines where to store the default command line and potential error output of your application as a text file.
# If no path is given your current working directory is used as the default
# You can redirect stderr messages into a separate file with -e <PATH>
#$ -o ./LogFiles

# Determines an e-mail address to send job information to.
# The options after -m control the cases in which to send a e-mail (b=begin, e=end, a=abort, s=suspend)
#$ -M lsobirey@physnet.uni-hamburg.de -m beas

# Influences job priority. Users can only decrease their job priority by setting a value between 0 and -1023.
#$ -p 0

# Request exclusive memory for your job. This value is automatically multiplied by the number of requested CPU slots.
# If not explicitly requested the default value is 1 GB. If your job exceeds the memory request it will be aborted.
#$ -l h_vmem=100G

# Request a maximum CPU time limit for your job.
# Note that the requested CPU time must be less than 1 hour in order to successfully submit to short.q.
# If not explicitly requested default value is 7 days. If your job exceeds the CPU time request it will be aborted.
#$ -l h_cpu=240:00:00

# FOR GPU APPLICATIONS ONLY:
# Request a specific GPU architecture. Your jobs will run on the given architecture only.
# Remove this if you want your jobs to run on all available GPUs regardless of the individual architecture.
# Available architectures in graphix.q are: 'titanxp', '1080ti' and '2080ti'
# Available architectures in graphit.q are: 'K20', 'K40' and 'K80'
# There are no GPUs available outside graphix.q or graphit.q queues.
#$ -l gpu_gen=3090

# Selects your current directory (where this job file is located) as a working directory for your job.
#$ -cwd

# Forces the user commands to be interpreted as bash commands
#$ -S /bin/bash


# USER COMMANDS IN BASH 
# This shebang incentivizes the following commands to be executed in a bash environment
#!/bin/bash

export OMP_NUM_THREADS=4
module load cuda/11.1.105
module load cudnn/8.0.5-cuda11.1
module load anaconda3/2021.05 
conda activate FermiUNvironment
python3 ./torchUnet.py
