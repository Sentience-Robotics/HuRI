resource "google_container_cluster" "primary" {
  name = var.cluster_name
  # Zonal (single zone) rather than regional: a regional cluster replicates every
  # node pool across all 3 zones, tripling SSD/CPU/GPU usage and blowing past
  # free-trial quotas. Zonal keeps one node per pool.
  location = var.zone

  # Allow `terraform destroy` to delete the cluster. The provider defaults this
  # to true, which blocks teardown — fine to disable for a throwaway trial stack.
  deletion_protection = false

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  # This default pool is created (then removed) during cluster creation, so it
  # must also fit the free-trial SSD quota. Without this it uses GKE's default
  # 100 GB pd-balanced (SSD) disk. pd-standard keeps it off SSD_TOTAL_GB.
  node_config {
    disk_type    = "pd-standard"
    disk_size_gb = 50
  }

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
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "e2-standard-4"

    # pd-standard (HDD) boot disk counts against DISKS_TOTAL_GB, not the much
    # smaller SSD_TOTAL_GB quota that pd-balanced/pd-ssd consume. Keeps the
    # free trial from tripping its 250 GB SSD limit.
    disk_type    = "pd-standard"
    disk_size_gb = 50

    # GKE reserves the node-role.kubernetes.io/ prefix and rejects it as a node
    # label, so we use a custom key to pin system pods (Ray head, Qdrant,
    # embedding, LiteLLM) here. Must match the nodeSelectors in values-gcp.yaml,
    # qdrant.tf, embedding.tf and litellm.tf.
    labels = {
      "huri.io/system" = "true"
    }

    # Removed taint to allow init jobs and other system pods to schedule easily

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# GPU node pool for NVIDIA workers
resource "google_container_node_pool" "gpu_nodes" {
  name     = "gpu-pool"
  location = var.zone
  cluster  = google_container_cluster.primary.name

  # Layer ③ — GKE provisions/removes GPU VMs as pending GPU pods appear/clear.
  # Only fires once KubeRay (layer ②, ray.enableInTreeAutoscaling in values)
  # marks worker pods Pending for lack of GPU. This is a zonal pool
  # (location = zone), so these counts are the actual node counts (no per-zone
  # multiplier).
  initial_node_count = var.gpu_min_nodes

  # Spread the pool across the region's zones (when gpu_node_zones is set) so the
  # autoscaler isn't trapped in one zone's GPU stock. Empty list -> nodes stay in
  # the cluster zone (var.zone). All listed zones must be within var.region.
  node_locations = length(var.gpu_node_zones) > 0 ? var.gpu_node_zones : null

  autoscaling {
    min_node_count = var.gpu_min_nodes
    max_node_count = var.gpu_max_nodes

    # ANY = provision in whichever listed zone currently has GPU capacity, rather
    # than the default BALANCED round-robin that keeps hitting the exhausted zone.
    # This is the knob that actually dodges RESOURCE_POOL_EXHAUSTED.
    location_policy = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  queued_provisioning {
    enabled = true
  }

  node_config {
    # machine_type and gpu_type must be compatible: N1 for T4, G2 for L4,
    # A2 for A100 (see variables.tf). guest_accelerator takes the *accelerator*
    # type (nvidia-tesla-t4, ...), never a machine type.
    machine_type = var.gpu_machine_type

    flex_start = true
    reservation_affinity {
      consume_reservation_type = "NO_RESERVATION"
    }

    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
    }

    # HDD boot disk to stay off the SSD_TOTAL_GB quota (see system-pool note).
    disk_type    = "pd-standard"
    disk_size_gb = 50

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

  lifecycle {
    # The autoscaler owns the running node count; don't let Terraform reset it
    # back to the initial count on every apply.
    ignore_changes = [node_count]
  }
}
