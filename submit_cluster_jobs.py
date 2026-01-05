import os
import time
import random

shell_script_name = f'temporary_shell_script.sh'
results_folder = 'experiment_results/random_vs_uncertain_vs_error_2'

num_of_seeds = 10
param_list = ['random', 'uncertain', 'error']

random_seeds = random.sample(range(100000000), num_of_seeds)

for param in param_list:
    for (seed_idx, seed) in enumerate(random_seeds):
        out_folder = os.path.join(results_folder, f"{param}_{seed_idx}")
        f = open(shell_script_name, 'x')
        f.write('#!/bin/bash \n')
        f.write('source /etc/profile \n')
        f.write('module load cuda/11.8 \n')
        f.write(f'python active_learning.py --final-budget=2500 --prioritizer={param} --active-batch-size=50 --seed={seed} --out={out_folder} --lr=0.1 --lambda-kl=1.0\n')
        f.close()
        os.system(f'LLsub {shell_script_name} -g volta:1 -o {results_folder}/output_{param}_{seed_idx}.log')
        os.remove(shell_script_name)
        time.sleep(10)