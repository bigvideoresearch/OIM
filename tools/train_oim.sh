#!/usr/bin/env bash

# Usage:
# ./tools/train_oim.sh oim_train models/VGG16/solver.prototxt 0.2 0.2 5.0 data/imagenet_models/$VGG16_model_name
# ./tools/train_oim.sh oim_train_ft models/VGG16/solver_ft.prototxt 0.2 0.2 2.0 output/oim_train/voc_2007_trainval/vgg16_oim_iter_70000.caffemodel


DATE=$(date +"%Y%m%d_%H%M%S")

run_name=${1}
solver=${2}
alpha=${3}
beta=${4}
ratio=${5}
weights=${6}

mkdir -p output/${run_name}
cp -rv models output/${run_name}

export alpha=${alpha}
export beta=${beta}
export ratio=${ratio}
echo "Running arguments: run_name=${run_name}, solver=${solver}, alpha=${alpha}, beta=${beta}, ratio=${ratio}, weights=${weights}" \
    | tee output/${run_name}/train_oim.log.${DATE}

./tools/train_net.py --gpu 0 --solver ${solver} \
    --weights ${weights} --iters 70000 \
    2>&1 | tee -a output/${run_name}/train_oim.log.${DATE}