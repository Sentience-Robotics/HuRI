#!/bin/bash

# Run when fully deployed

echo "Port-Forwarding to HuRI's dashboard and server..."

if ! kubectl get services | grep huri-service-head-svc >/dev/null 2>&1; then
    echo "HuRI is not fully deployed"
    exit 1
fi

kubectl port-forward service/huri-service-head-svc 8265:8265 & # dashboard
kubectl port-forward service/huri-service-head-svc 8000:8000 # serve (blocking)
