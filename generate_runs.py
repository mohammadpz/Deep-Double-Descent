ks = [4, 10, 20, 60]
noises = [0.15]

names = []

for k in ks:
    for noise in noises:
        name = 'noise_' + str(noise) + 's_' + str(k)
        names += [name]
        run_file = open(name + ".sh", "w")
        run_file.write('#!/bin/bash\n')
        run_file.write('~/miniconda/bin/python train_cifar10.py ' + str(k) + ' ' + str(noise) + ' >> ' + str(k) + '_' + str(noise) + '.txt')
        run_file.flush()

run_all = open("run_all.sh", "w")
run_all.write('#!/bin/bash\n')
for name in names:
    run_all.write('sbatch --time=10:00:0 --gres=gpu:1 --mem=12G --account=rrg-bengioy-ad_gpu '
                  '--cpus-per-task=4 --output=/home/pezeshki/scratch/logs/slurm-%j.out '
                  '/home/pezeshki/scratch/dd/Deep-Double-Descent/' +
                  name + ".sh\n")
run_all.write('watch -n 0.1 squeue -u pezeshki')
run_all.flush()
