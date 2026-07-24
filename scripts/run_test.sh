#!/bin/bash

eval "$(conda shell.bash hook)"


conda activate imperio


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

ablations=("0.1,20" "1,5" "1,10" "1,50" "5,20" "10,20" "wo_teb" "wo_ceb" "wo_clma")
path="ablation/"

for ablation in "${ablations[@]}"
do
    echo "正在测试 ablation: $ablation"
    python test_LLVIP.py --model_path $path$ablation
    echo "--------------------------"
done