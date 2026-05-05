#!/bin/bash

# Run when src/ changes

echo "Building HuRI's image..."

docker build -f deploy/Dockerfile -t huri-ray-image:latest . || true

if [[ "$1" == "--local" ]]; then
    # load image in the kind local cluster
    kind load docker-image huri-ray-image:latest || true
else; then
    echo "Only --local is supported right now (still figuring it out)"
    exit 1
fi
