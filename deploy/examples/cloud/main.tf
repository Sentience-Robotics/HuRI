provider "google" {
  project = var.project_id
  region  = var.region
}

# Get GKE cluster credentials
data "google_client_config" "default" {}

data "google_container_cluster" "primary" {
  name       = google_container_cluster.primary.name
  location   = var.zone
  depends_on = [google_container_cluster.primary]
}

provider "kubernetes" {
  host                   = "https://${data.google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(data.google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes = {
    host                   = "https://${data.google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(data.google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
  }
}

# Install KubeRay Operator
resource "helm_release" "kuberay_operator" {
  name             = "kuberay-operator"
  repository       = "https://ray-project.github.io/kuberay-helm/"
  chart            = "kuberay-operator"
  namespace        = "ray-system"
  create_namespace = true
}

# Install HuRI App
resource "helm_release" "huri" {
  name             = "huri"
  chart            = "../../../helm"
  namespace        = "huri"
  create_namespace = true

  values = [
    file("${path.module}/values-gcp.yaml")
  ]

  depends_on = [
    helm_release.kuberay_operator,
    google_container_node_pool.gpu_nodes,
    google_container_node_pool.system_nodes,
    # RAGHandle reads these at startup via values-gcp.yaml user_config.
    helm_release.qdrant,
    kubernetes_service_v1.litellm,
    kubernetes_service_v1.embedding
  ]
}
