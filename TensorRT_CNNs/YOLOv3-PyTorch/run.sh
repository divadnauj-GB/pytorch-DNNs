#!/bin/bash
APP_ARGS=$*
NEW_VAR=("$@")
rm -r ${APP_DIR}/results/inference/results || true
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate Pytorch_nvbitPERfi

# python Run_Layer.py -n LeNet -ln 1 -bs 1 -onnx
# python Run_Layer.py -n LeNet -ln 0 -bs 1 -trt -sz 1 6 28 28

eval ${PRELOAD_FLAG} python3 ${BIN_DIR}/${APP_BIN} ${APP_ARGS} > stdout.txt 2> stderr.txt 

echo "###################################################" > ${APP_DIR}/results/inference/results.csv

PATH_YOLOv3=${APP_DIR}/results/inference/results

for file in `ls ${PATH_YOLOv3}/* | sort -n -t _ -k 2`; do
    final=${file#"${APP_DIR}/results/inference/results/"};
    echo ${final}  >> ${APP_DIR}/results/inference/results.csv 
    cat ${file} >> ${APP_DIR}/results/inference/results.csv 
done
#for ((idx=0; idx<${#NEW_VAR[@]}; ++idx))
#do
#    if [ "${NEW_VAR[idx]}" == "-t" ] 
#    then
#        TYPE="${NEW_VAR[idx+1]}"
#
#    elif [ "${NEW_VAR[idx]}" == "-n" ] 
#    then
#        NAME="${NEW_VAR[idx+1]}"
#
#    elif [ "${NEW_VAR[idx]}" == "-ln" ] 
#    then
#        NUM="${NEW_VAR[idx+1]}"
#    fi
#done

if [ $GOLDEN_FLAG ]
then
    rm -r ${APP_DIR}/results/inference/golden_results || true
    rm ${APP_DIR}/results/inference/golden_results.csv || true
    mv ${APP_DIR}/results/inference/results ${APP_DIR}/results/inference/golden_results || true
    mv ${APP_DIR}/results/inference/results.csv ${APP_DIR}/results/inference/golden_results.csv || true
fi

