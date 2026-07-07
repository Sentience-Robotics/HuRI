output "kubernetes_cluster_name" {
  value       = google_container_cluster.primary.name
  description = "GKE Cluster Name"
}

output "kubernetes_cluster_host" {
  value       = google_container_cluster.primary.endpoint
  description = "GKE Cluster Host"
}

output "region" {
  value       = var.region
  description = "GCP region"
}

output "project_id" {
  value       = var.project_id
  description = "GCP Project ID"
}

output "qdrant_url" {
  value       = "http://qdrant.huri.svc.cluster.local:6333"
  description = "In-cluster Qdrant endpoint (matches values-gcp.yaml)"
}

output "llm_url" {
  value       = "http://litellm.huri.svc.cluster.local:4000"
  description = "In-cluster LiteLLM (Gemini) endpoint (matches values-gcp.yaml)"
}
