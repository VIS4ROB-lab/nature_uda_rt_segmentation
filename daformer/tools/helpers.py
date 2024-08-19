import subprocess


def fetch_experiment(name: str) -> None:
    subprocess.run([
        'rsync', '-a',
        f'v4rl:/home/han/uda_natural_envs_wang/daformer/work_dirs/local-basic/{name}',
        'work_dirs/local-basic/'
    ])
