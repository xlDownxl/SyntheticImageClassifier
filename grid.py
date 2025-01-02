import itertools
import os
import subprocess

#Grid of parameters, in this case I tested only the impact of pretraining the resnet18 on imagenet and whether the highpass filter improves the result
parameter_grid = {
    'batch_size': [32],
    'lr': [0.0001],
    'warmup_epochs': [1],
    'use_fourier': [True],
    'crop': [True],
    'pretrained': [True, False],
    'apply_highpass': [True, False],
    'highpass_cutoff': [10],

}

# Generate all combinations of parameters
keys, values = zip(*parameter_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

jobs_dir = "jobs"
os.makedirs(jobs_dir, exist_ok=True)

# Create .job files and submit jobs
for idx, combination in enumerate(combinations):
    # Generate the parameter string
    param_str = " ".join(
        [
            f"--{key} {value}" if not isinstance(value, bool) else (f"--{key}" if value else "")
            for key, value in combination.items()
        ]
    ).strip() 
    #print(param_str)

    # Generate a unique filename for the job
    job_name = f"run_{idx}_{'_'.join([str(v).replace('/', '').replace('[', '').replace(']', '').replace(' ', '_') for v in combination.values() if not isinstance(v, bool) or v])}"
    job_file = os.path.join(jobs_dir, f"{job_name}.job")

    # Define the job file content
    job_content = f"""#!/bin/bash
#SBATCH --job-name=grid_{idx}
#SBATCH --chdir=/wrk-kappa/users/nicjosw/SyntheticImageClassifier/jobs
#SBATCH -o job_{idx}-{job_name}-%j.txt
#SBATCH -p gpu
#SBATCH -M kale
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 1-0
#SBATCH --constraint=v100

module purge                             
module load Python/3.9.6-GCCcore-11.2.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
cd /wrk-kappa/users/nicjosw
source  torch-3.9/bin/activate
cd SyntheticImageClassifier

mkdir /wrk-kappa/users/nicjosw/tmp/$SLURM_JOB_ID/
mkdir /wrk-kappa/users/nicjosw/tmp/$SLURM_JOB_ID/$(date +"%d-%m-%Y-%H-%M-%S")
export TMPDIR=/wrk-kappa/users/nicjosw/tmp/$SLURM_JOB_ID/$(date +"%d-%m-%Y-%H-%M-%S")

srun python main.py {param_str} --num_epochs 10 --img_size 640 480 --num_workers 12
"""

    # Write the job file
    with open(job_file, "w") as f:
        f.write(job_content)

    # Submit the job
    subprocess.run(["sbatch", job_file])

print(f"Generated and submitted {len(combinations)} jobs.")
