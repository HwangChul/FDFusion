#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."


ablations=("wo_teb" "wo_ceb" "wo_clma")
path="model/"

for ablation in "${ablations[@]}"
do
    echo "正在测试 ablation: $ablation"
    python fusion_LLVIP.py --ablation "$ablation" --model_path $path$ablation
    echo "--------------------------"
done

model_paths=("model/0.1,20" "model/1,5" "model/1,10" "model/1,50" "model/5,20" "model/10,20")

for model_path in "${model_paths[@]}"
do
    echo "正在测试 model_path: $model_path"
    python fusion_LLVIP.py --model_path "$model_path"
    echo "--------------------------"
done

/usr/bin/shutdown -h now