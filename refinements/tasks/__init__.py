# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import os


def get_task_names():
    task_names = [f[:-3] for f in os.listdir(os.path.dirname(__file__)) if f.endswith('.py') and f != '__init__.py']
    return task_names
