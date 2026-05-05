#!/bin/bash

# Run when you want to stop everything

if [[ "$1" == "--clean" ]]; then
    deploy/clean.sh
fi


echo "Deploying HuRI..."

if [[ "$2" == "--local" ]]; then
    # Kind will mimic a cluster locally
    kind create cluster --image=kindest/node:v1.26.0 || true
elif ! kubectl get nodes >/dev/null 2>&1; then
  echo "Kubernetes cluster is not reachable"
  exit 1
fi


helm install kuberay-operator kuberay/kuberay-operator --version 1.6.0 || true
kubectl apply -f deploy/huri_service.yaml
