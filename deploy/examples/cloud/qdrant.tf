# Qdrant vector database — backs the HuRI RAG store.
#
# Deployed in-cluster so the RAGHandle reaches it over the cluster network at
# http://qdrant.huri.svc.cluster.local:6333 (see values-gcp.yaml user_config),
# replacing the old external qdrant.pommier.lan host. make_qdrant_client()
# derives port 6333 for non-https URLs, so no app change is needed.

resource "helm_release" "qdrant" {
  name       = "qdrant"
  repository = "https://qdrant.github.io/qdrant-helm"
  chart      = "qdrant"
  # Pinned so the chart (and the Qdrant image it carries) don't float on apply.
  # Chart 1.18.2 → appVersion v1.18.1. Bump deliberately, not implicitly.
  version          = "1.18.2"
  namespace        = "huri"
  create_namespace = true

  # Single-node, persistent. The chart's StatefulSet exposes a ClusterIP
  # Service named "qdrant" on the REST (6333) and gRPC (6334) ports.
  values = [
    yamlencode({
      replicaCount = 1
      persistence = {
        size             = "10Gi"
        storageClassName = "standard-rwo"
      }
      service = {
        type = "ClusterIP"
      }
      # Keep Qdrant on the system pool (CPU only); the GPU pool is reserved
      # for the Ray workers.
      nodeSelector = {
        "huri.io/system" = "true"
      }
      resources = {
        requests = { cpu = "250m", memory = "512Mi" }
        limits   = { cpu = "1", memory = "2Gi" }
      }
    })
  ]
}
