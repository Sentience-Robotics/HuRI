# LiteLLM proxy — a fast, OpenAI-compatible LLM endpoint backed by Mistral.
#
# The RAGHandle (values-gcp.yaml user_config) points llm_url at
# http://litellm.huri.svc.cluster.local:4000 with llm_provider: vllm and
# llm_model: huri-fast. LiteLLM exposes /v1/chat/completions (with streaming),
# which is exactly the path the handler builds, and forwards to Mistral's
# OpenAI-compatible API.

# Mistral API key, read from GCP Secret Manager (secrets.tf), never committed.
resource "kubernetes_secret_v1" "litellm" {
  metadata {
    name      = "litellm-secrets"
    namespace = "huri"
  }
  data = {
    # trimspace() guards against a trailing newline/CRLF in the Secret Manager
    # value (e.g. created with `echo` instead of `printf`). LiteLLM puts this
    # key straight into the outbound Authorization header, and aiohttp rejects
    # any header value containing \r or \n ("header injection"), which surfaces
    # as a 500 back to the RAG caller.
    MISTRAL_API_KEY = trimspace(data.google_secret_manager_secret_version.mistral_api_key.secret_data)
  }

  # The huri namespace is created by the qdrant release; reuse it.
  depends_on = [helm_release.qdrant]
}

# model_list config, with the Mistral model name templated in.
resource "kubernetes_config_map_v1" "litellm" {
  metadata {
    name      = "litellm-config"
    namespace = "huri"
  }
  data = {
    "config.yaml" = templatefile("${path.module}/litellm-config.yaml", {
      model = var.llm_model
    })
  }

  depends_on = [helm_release.qdrant]
}

resource "kubernetes_deployment_v1" "litellm" {
  metadata {
    name      = "litellm"
    namespace = "huri"
    labels    = { app = "litellm" }
  }

  spec {
    replicas = 1
    selector {
      match_labels = { app = "litellm" }
    }
    template {
      metadata {
        labels = { app = "litellm" }
      }
      spec {
        # CPU-only proxy — keep it off the GPU pool.
        node_selector = {
          "huri.io/system" = "true"
        }

        container {
          name = "litellm"
          # Pinned by digest for reproducibility; the tag is kept for readability.
          # Re-resolve with: docker buildx imagetools inspect ghcr.io/berriai/litellm:main-stable
          image = "ghcr.io/berriai/litellm:main-stable@sha256:8f3517476b8293ccc1c3b0da7940d8dabbfe64b013454113f5a3cb68265c439b"
          args  = ["--config", "/etc/litellm/config.yaml", "--port", "4000"]

          port {
            container_port = 4000
          }

          env_from {
            secret_ref {
              name = kubernetes_secret_v1.litellm.metadata[0].name
            }
          }

          volume_mount {
            name       = "config"
            mount_path = "/etc/litellm"
            read_only  = true
          }

          resources {
            # The litellm:main-stable image loads many provider SDKs at startup
            # and OOMs under a 1Gi cap before it can serve /health. 2Gi is the
            # observed safe ceiling for the proxy.
            requests = { cpu = "100m", memory = "512Mi" }
            limits   = { cpu = "500m", memory = "2Gi" }
          }

          readiness_probe {
            http_get {
              path = "/health/liveliness"
              port = 4000
            }
            initial_delay_seconds = 10
            period_seconds        = 10
          }
        }

        volume {
          name = "config"
          config_map {
            name = kubernetes_config_map_v1.litellm.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "litellm" {
  metadata {
    name      = "litellm"
    namespace = "huri"
  }
  spec {
    selector = { app = "litellm" }
    port {
      port        = 4000
      target_port = 4000
    }
    type = "ClusterIP"
  }

  # The huri namespace is created by the qdrant release; reuse it. Without this
  # the service races ahead of namespace creation ("namespaces huri not found").
  depends_on = [helm_release.qdrant]
}
