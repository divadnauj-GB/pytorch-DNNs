#!/bin/bash
APP_ARGS=$*
NEW_VAR=("$@")

#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate Pytorch_nvbitPERfi

# python Run_Layer.py -n LeNet -ln 1 -bs 1 -onnx
# python Run_Layer.py -n LeNet -ln 0 -bs 1 -trt -sz 1 6 28 28

eval ${PRELOAD_FLAG} python3 ${BIN_DIR}/${APP_BIN} ${APP_ARGS} > stdout.txt 2> stderr.txt


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
    for ((idx=0; idx<${#NEW_VAR[@]}; ++idx))
    do  
        case "${NEW_VAR[idx]}" in

        "-t")
            TYPE="${NEW_VAR[idx+1]}"
            ;;

        "-n")
            NAME="${NEW_VAR[idx+1]}"
            ;;

        "-ln")
            NUM="${NEW_VAR[idx+1]}"
            ;;

        *)
        ;;
        esac
    done

    mv ${APP_DIR}/${TYPE}/${NAME}/Outputs_DNN.h5 ${APP_DIR}/${TYPE}/${NAME}/Golden_Outputs_DNN.h5 

fi

