#!/usr/bin/env bash
set -e

# Usage: ./tools/test_oim.sh oim_test models/VGG16/test.prototxt output/oim_train_ft/voc_2007_trainval/vgg16_oim_ft_iter_20000.caffemodel

DATE=$(date +"%Y%m%d_%H%M%S")

run_name=${1}
prototxt=${2}
caffemodel=${3}

mkdir -p output/${run_name}
echo "caffemodel: $(pwd), ${caffemodel}" | tee output/${run_name}/test_oim.log.${DATE}

./tools/test_net.py --run_name ${run_name} --gpu 0 --def ${prototxt} \
    --net ${caffemodel} --imdb voc_2007_test 2>&1 | tee -a output/${run_name}/test_oim.log.${DATE}
