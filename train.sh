#!/bin/bash
if ! test -f example/MobileNetSSD_train.prototxt ;then
        echo "error: example/MobileNetSSD_train.prototxt does not exist."
        echo "please use the gen_model.sh to generate your own model."
        exit 1
fi

if [ ! -d "snapshot" ];then
        echo "snapshot folder does not exist."
        echo "using the default 73000_iter model"
        mkdir -p snapshot
        ../../build/tools/caffe train -solver="solver_train.prototxt" \
        -weights="mobilenet_iter_73000.caffemodel" \
        -gpu 0,1 2>&1 | tee /home/$USER/caffe_mobilenet_training.log
else
        echo "snapshot folder exists"
        max=-1
        for file in snapshot/*.caffemodel
        do
                num=${file:24}
                num=${num%.caffemodel}
                echo "$num"
                [ "$num" -gt "$max" ] && max=$num
        done
        echo "latest snapshot model is: $max"
        ../../build/tools/caffe train -solver="solver_train.prototxt" \
        -weights="snapshot/mobilenet_iter_$max.caffemodel" \
        -gpu 0,1 2>&1 | tee /home/$USER/caffe_mobilenet_training.log
fi
