#!/bin/bash

nohup python run_experiments.py --config configs/ablation/farm_sim2real_uda_rcs_copypaste_load_daformer.py > exp1.log 2>&1 &

while [ $(ps -p $! -o comm=) ]
do
    sleep 1
done

nohup python run_experiments.py --config configs/daformer/b1.py > exp2.log 2>&1 &

#nohup python run_experiments.py --config configs/temp/resume/fast_1450.py > exp2.log 2>&1 &
#
#while [ $(ps -p $! -o comm=) ]
#do
#   sleep 1
#done
#
#nohup python run_experiments.py --config configs/temp/resume/norm_900.py > exp3.log 2>&1 &
#
#while [ $(ps -p $! -o comm=) ]
#do
#   sleep 1
#done
#
#nohup python run_experiments.py --config configs/temp/resume/norm_1450.py > exp4.log 2>&1 &