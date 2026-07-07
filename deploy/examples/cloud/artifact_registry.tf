# Private Artifact Registry holding the website backend image.
#
# This repository ALREADY EXISTS — Terraform should adopt it rather than try to
# recreate it. After `terraform init`, import it once:
#
#   terraform import google_artifact_registry_repository.website \
#     projects/${PROJECT}/locations/${LOCATION}/repositories/${REPO_ID}
#
# (LOCATION = var.website_image_location, REPO_ID = var.website_image_repo.)
# The image itself is built and pushed out-of-band from
# HuRI_website_demo/backend/Dockerfile.
resource "google_artifact_registry_repository" "website" {
  location      = var.website_image_location
  repository_id = var.website_image_repo
  format        = "DOCKER"
  description   = "HuRI website backend images"

  # Don't let an apply delete an existing shared registry.
  lifecycle {
    prevent_destroy = true
  }
}

data "google_project" "current" {}

# GKE nodes pull with the node pool's service account (the default Compute
# Engine SA here). Grant it read access to the repo so the kubelet can pull.
resource "google_artifact_registry_repository_iam_member" "website_puller" {
  project    = var.project_id
  location   = google_artifact_registry_repository.website.location
  repository = google_artifact_registry_repository.website.repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"
}
