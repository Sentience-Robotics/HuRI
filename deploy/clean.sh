#!/bin/bash

# Run when you want to stop everything

echo "Stopping and cleaning HuRI..."

kubectl delete rayservice --all
helm uninstall kuberay-operator

kind delete cluster
