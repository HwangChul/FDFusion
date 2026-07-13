#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

epochs=("20" "30" "40" "50" "60" "70")

for epoch in "${epochs[@]}"
do
    echo "正在测试 Epoch: $epoch"
    python test_LLVIP.py --pth_epoch "$epoch"
    echo "--------------------------"
done