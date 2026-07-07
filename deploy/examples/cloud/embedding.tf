# Embedding server — serves bge-large-en-v1.5 over an OpenAI-compatible
# /v1/embeddings endpoint, in-cluster, replacing the old external
# embedding.huri.lan host.
#
# Why in-cluster and not a managed GCP/Vertex endpoint:
# RAG retrieval embeds the query and searches Qdrant by vector similarity, so
# query vectors MUST come from the SAME model that produced the stored vectors
# at ingestion time. Those were created with the bge-large-en-v1.5 GGUF
# (Q4_K_M, 1024-dim). Vertex AI only serves its own embedding models (different
# vectors and dimensions), which would be incompatible without re-ingesting the
# whole corpus. So we run the exact same GGUF here via llama.cpp's server, whose
# /v1/embeddings path and request shape match RemoteEmbedder (ingestion.py) and
# RAGHandle._embed (rag.py).
#
# The RAGHandle (values-gcp.yaml user_config) points embedding_url at
# http://embedding.huri.svc.cluster.local:8080. llama.cpp ignores the request's
# "model" field and uses the loaded GGUF, so embedding_model is informational.

resource "kubernetes_deployment_v1" "embedding" {
  metadata {
    name      = "embedding"
    namespace = "huri"
    labels    = { app = "embedding" }
  }

  spec {
    replicas = 1
    selector {
      match_labels = { app = "embedding" }
    }
    # The model cache is a ReadWriteOnce PVC, so the old pod must release it
    # before the new one mounts it — a rolling update would deadlock.
    strategy {
      type = "Recreate"
    }
    template {
      metadata {
        labels = { app = "embedding" }
      }
      spec {
        # CPU-only — bge-large (~335M params, ~200MB at Q4_K_M) runs fine on the
        # system pool. The GPU pool is reserved for the Ray workers.
        node_selector = {
          "huri.io/system" = "true"
        }

        container {
          name = "embedding"
          # Pinned by digest for reproducibility; the tag is kept for readability.
          # Re-resolve with: docker buildx imagetools inspect ghcr.io/ggml-org/llama.cpp:server
          # IfNotPresent (also the default for a non-:latest tag) reuses the image
          # from the node cache across restarts; only re-pulled if the node is replaced.
          image = "ghcr.io/ggml-org/llama.cpp:server@sha256:9e2ab5775c2e9cb52ac611e506ea5b873247033aeba54d71f20d71bf7f4e219d"
          # -hf pulls the GGUF from Hugging Face on startup into HF_HOME.
          # --pooling cls matches how bge models are meant to be pooled.
          # -c 512 covers bge's 512-token context (ingestion chunks stay under
          # this — see the rag-ingestion-chunk-size note).
          args = [
            "-hf", var.embedding_hf_repo,
            "--embedding",
            "--pooling", "cls",
            "--host", "0.0.0.0",
            "--port", "8080",
            "-c", "512",
          ]

          env {
            name  = "HF_HOME"
            value = "/models"
          }

          port {
            container_port = 8080
          }

          volume_mount {
            name       = "models"
            mount_path = "/models"
          }

          resources {
            requests = { cpu = "250m", memory = "1Gi" }
            limits   = { cpu = "2", memory = "2Gi" }
          }

          # The model is downloaded at startup, so give it generous startup slack
          # before the readiness gate kicks in.
          readiness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 30
            period_seconds        = 10
            failure_threshold     = 30
          }
        }

        # Persistent model cache so the GGUF is downloaded once and survives pod
        # restarts/reschedules (emptyDir would re-pull it every time).
        volume {
          name = "models"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim_v1.embedding_models.metadata[0].name
          }
        }
      }
    }
  }

  # The huri namespace is created by the qdrant release; reuse it.
  depends_on = [helm_release.qdrant]
}

resource "kubernetes_persistent_volume_claim_v1" "embedding_models" {
  metadata {
    name      = "embedding-models"
    namespace = "huri"
  }
  # standard-rwo binds WaitForFirstConsumer, so the PVC stays Pending until the
  # pod mounts it. Don't block apply waiting for a bind that can't happen yet.
  wait_until_bound = false

  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = "standard-rwo"
    resources {
      requests = { storage = "1Gi" } # models shouldn't be that big
    }
  }

  # The huri namespace is created by the qdrant release; reuse it.
  depends_on = [helm_release.qdrant]
}

resource "kubernetes_service_v1" "embedding" {
  metadata {
    name      = "embedding"
    namespace = "huri"
  }
  spec {
    selector = { app = "embedding" }
    port {
      port        = 8080
      target_port = 8080
    }
    type = "ClusterIP"
  }

  # The huri namespace is created by the qdrant release; reuse it. Without this
  # the service races ahead of namespace creation ("namespaces huri not found").
  depends_on = [helm_release.qdrant]
}
