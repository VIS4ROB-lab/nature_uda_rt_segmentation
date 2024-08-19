"""
Helper function to remove a prefix from keys in a state dict of a torch checkpoint.
Indicate whether to keep other keys or not.
"""

# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import argparse
from collections import OrderedDict
from typing import Dict
import torch


def remove_prefix(ckpt: Dict, prefix: str, keep_others: bool = False) -> Dict:
    new_ckpt = {'state_dict': OrderedDict()}
    for k, v in ckpt['state_dict'].items():
        if k.startswith(prefix):
            new_ckpt['state_dict'][k[len(prefix)+1:]] = v
        elif keep_others:
            new_ckpt['state_dict'][k] = v
    return new_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove prefix from keys in a state dict.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input state dict")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output state dict")
    parser.add_argument("-p", "--prefix", required=True, type=str, help="Prefix to remove")
    parser.add_argument("--keep_others", action="store_true", help="Keep other keys")

    args = parser.parse_args()
    in_dict = torch.load(args.input, map_location="cpu")
    out_dict = remove_prefix(in_dict, args.prefix, args.keep_others)
    torch.save(out_dict, args.output)
