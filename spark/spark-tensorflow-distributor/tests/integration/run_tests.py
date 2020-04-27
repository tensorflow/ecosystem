import subprocess
import argparse


MAX_NUM_WORKERS = 2


subprocess.run(
    [
        'docker-compose down && '
        'docker-compose up -d --scale worker={max_num_workers} && '
        'docker-compose exec -T master python -m pytest -s tests/integration '
        '--max_num_workers {max_num_workers} && '
        'docker-compose down'.format(max_num_workers=MAX_NUM_WORKERS)
    ],
    shell=True,
    check=True,
)
