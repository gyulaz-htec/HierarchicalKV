#!/bin/bash

docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size=16G \
    -v "$(pwd)":/workspace/code \
    -w /workspace/code \
    rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.9.1