resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  release_channel {
    channel = "REGULAR"
  }

  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# General purpose node pool for Ray Head and other system components
resource "google_container_node_pool" "system_nodes" {
  name       = "system-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "e2-standard-4"
    
    labels = {
      "node-role.kubernetes.io/control-plane" = "true"
    }

    # Removed taint to allow init jobs and other system pods to schedule easily
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# GPU node pool for NVIDIA workers
resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "n1-standard-4" # T4 requires N1 or G2. For L4, use G2.
    
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
    }

    # Automatically install NVIDIA drivers
    # This is recommended for GKE
    # metadata = {
    #   "install-nvidia-driver" = "true"
    # }

    labels = {
      gpu = "nvidia"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}
