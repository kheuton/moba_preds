#!/usr/bin/env bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-04:00          # Runtime in D-HH:MM
#SBATCH -p preempt            # Partition to submit to
#SBATCH --mem-per-cpu 20000  # Memory (in MB) per cpu
#SBATCH -o /cluster/tufts/hugheslab/kheuto01/slurmlog/out/log_%j.out       # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/kheuto01/slurmlog/err/log_%j.err       # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children


python  test_run.py $args

