# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------

import subprocess
import argparse
import pathlib

parser = argparse.ArgumentParser(description='Run a tools/inference with an experiment')
parser.add_argument('exp', type=str, help='Experiment folder name in ./work_dirs/local-basic/')
parser.add_argument('--iter', type=str, default='',
                    help='Checkpoint file to load. Default (empty): latest. Example: 9000 -> load iter_9000.pth')
parser.add_argument('--image_dir', type=str, default='data/Apple_Farm_Sim/images/validation',)
parser.add_argument("--timing", action='store_true', help="Perform timing test. Ignore all outputs")
parser.add_argument("--save_npy", action='store_true', help="Save npy files instead of images")
parser.add_argument("--save_dir", type=str, default='', help="Path to save the predictions")
parser.add_argument("--save_conf", action='store_true',
                    help="Save confidence of each pixel. Should be used with --save_npy")
parser.add_argument("--sky_only", action='store_true', help="Save only the sky mask")
parser.add_argument("--save_png", action='store_true', help="Save the original png")
parser.add_argument("--save_img", action='store_true', help="Save the overlaid images")
args = parser.parse_args()
exp = args.exp
if exp.endswith('/'):
    exp = exp[:-1]
it = 'iter_' + args.iter if len(args.iter) else 'latest'
if args.save_dir == '':
    args.save_dir = f'work_dirs/local-basic/{exp}/inference'

if not pathlib.Path('work_dirs/local-basic/' + exp).exists():
    raise ValueError(f"Experiment {exp} not found!")


if args.timing:
    subprocess.run([
        'python3', '-m', 'tools.inference',
        f'work_dirs/local-basic/{exp}/{exp}.json', f'work_dirs/local-basic/{exp}/{it}.pth',
        '--image', args.image_dir,
        '--timing'
    ])
    exit(0)
else:
    assert args.save_npy or args.sky_only or args.save_png or args.save_img, "Nothing to save!"
    if args.save_img:
        subprocess.run([
            'python3', '-m', 'tools.inference',
            f'work_dirs/local-basic/{exp}/{exp}.json', f'work_dirs/local-basic/{exp}/{it}.pth',
            '--image', args.image_dir,
            '--out_dir', args.save_dir,
            '--opacity', '0.6'
        ])
    elif args.sky_only:
        subprocess.run([
            'python3', '-m', 'tools.inference',
            f'work_dirs/local-basic/{exp}/{exp}.json', f'work_dirs/local-basic/{exp}/{it}.pth',
            '--image', args.image_dir,
            '--max_images', '-1',
            '--out_dir', args.save_dir,
            '--sky_only'
        ])
    elif args.save_npy:
        commands = [
            'python3', '-m', 'tools.inference',
            f'work_dirs/local-basic/{exp}/{exp}.json', f'work_dirs/local-basic/{exp}/{it}.pth',
            '--image', args.image_dir,
            '--max_images', '-1',
            '--out_dir', args.save_dir,
            '--save_npy'
        ]
        if args.save_conf:
            commands.append('--save_conf')
        subprocess.run(commands)
    elif args.save_png:
        subprocess.run([
            'python3', '-m', 'tools.inference',
            f'work_dirs/local-basic/{exp}/{exp}.json', f'work_dirs/local-basic/{exp}/{it}.pth',
            '--image', args.image_dir,
            '--out_dir', args.save_dir,
            '--max_images', '-1',
            '--raw_png'
        ])