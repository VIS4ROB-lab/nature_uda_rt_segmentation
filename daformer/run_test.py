# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------

import subprocess
import argparse
import pathlib

parser = argparse.ArgumentParser(description='Run a tool.test with an experiment')
parser.add_argument('exp', type=str, help='Experiment folder name in ./work_dirs/local-basic/')
parser.add_argument('mode', type=str, choices=['eval', 'logit', 'pred'],
                    help='eval, save_pred or save_logit mode')
parser.add_argument('--iter', type=str, default='',
                    help='Checkpoint file to load. Default (empty): latest. Example: 9000 -> load iter_9000.pth')
parser.add_argument("--out", action="store_true",
                    help="whether to save output")

args = parser.parse_args()
exp = args.exp
if exp.endswith('/'):
    exp = exp[:-1]
it = 'iter_' + args.iter if len(args.iter) else 'latest'

if not pathlib.Path('work_dirs/local-basic/' + exp).exists():
    print(f"Fetching experiment {args.exp} from training server")
    subprocess.run([
        'rsync', '-a',
        f'v4rl:/home/han/uda_natural_envs_wang/daformer/work_dirs/local-basic/{args.exp}',
        'work_dirs/local-basic/'
    ])

if args.mode == 'eval':
    print("Evaluating...")
    result = subprocess.run([
        'python3', '-m', 'tools.test',
        f'work_dirs/local-basic/{exp}/{exp}.json', f'work_dirs/local-basic/{exp}/{it}.pth',
        '--eval', 'mIoU', 'mDice',
        '--show', '--show-dir', f'work_dirs/local-basic/{exp}/out',
        '--opacity', '0.6'],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True)
    with open(f'work_dirs/local-basic/{exp}/out.txt', 'w') as f:
        f.write(result.stdout)
    print(f"Experiment {exp} Results:")
    subprocess.run([
        'tail', '-n', '18', f'work_dirs/local-basic/{exp}/out.txt'
    ])
elif args.mode == 'pred':
    subprocess.run([
        'python3', '-m', 'tools.test',
        f'work_dirs/local-basic/{exp}/{exp}.json', f'work_dirs/local-basic/{exp}/{it}.pth',
        '--save_pred',
        '--show-dir', f'work_dirs/local-basic/{exp}/out',
    ])
elif args.mode == 'logit':
    subprocess.run([
        'python3', '-m', 'tools.test',
        f'work_dirs/local-basic/{exp}/{exp}.json', f'work_dirs/local-basic/{exp}/{it}.pth',
        '--save_logit',
        '--show-dir', f'work_dirs/local-basic/{exp}/out',
    ])
