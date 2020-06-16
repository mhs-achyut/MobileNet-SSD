#!/bin/sh
if ! test -f example/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi

if ! test -f ../snapshot/*.caffemodel ;then
	echo "snapshot folder does not exist."
	echo "using the default 73000_iter model"
	mkdir -p snapshot
	../../build/tools/caffe train -solver="solver_train.prototxt" \
	-weights="mobilenet_iter_73000.caffemodel" \
	-gpu 0 2>&1 | tee /home/$USER/caffe_mobilenet_training.log
else
	unset -v latest
	for file in "snapshot"/*; do
	  [[ $file -nt $latest ]] && latest=$file
	done
	../../build/tools/caffe train -solver="solver_train.prototxt" \
	-weights="$latest" \
	-gpu 0 2>&1 | tee /home/$USER/caffe_mobilenet_training.log
fi
