#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

int_wights=("5" "10")
grad_wights=("5" "10" "50")

for int_wight in "${int_wights[@]}"
do
    echo "超参数α int_wight: $int_wight"
    python train_LLVIP.py --int_wight "$int_wight" --grad_wight 20 --module "$int_wight, 20"
    echo "--------------------------"
done

for grad_wight in "${grad_wights[@]}"
do
    echo "超参数β grad_wight: $grad_wight"
    python train_LLVIP.py --int_wight 1 --grad_wight "$grad_wight" --module "1, $grad_wight"
    echo "--------------------------"
done

/usr/bin/shutdown -h now