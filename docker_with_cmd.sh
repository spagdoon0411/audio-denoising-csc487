sudo docker run --gpus all --rm -it -p 8888:8888 -v $PWD:/tf -w /tf tensorflow/tensorflow:latest-gpu $1
