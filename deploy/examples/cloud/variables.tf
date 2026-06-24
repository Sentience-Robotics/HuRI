variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region. Used for regional resources (VPC, Artifact Registry). The GKE cluster itself is zonal (see var.zone)."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone the GKE cluster and its node pools run in. A zonal cluster places one node per pool (vs. one per zone for a regional cluster), which keeps SSD/CPU/GPU usage within free-trial quotas. Must be a zone within var.region."
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "The name of the GKE cluster"
  type        = string
  default     = "huri-ray-cluster"
}

variable "gpu_type" {
  description = "GPU *accelerator* type attached to each node — NOT a machine type. E.g. nvidia-tesla-t4, nvidia-l4, nvidia-tesla-a100."
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_machine_type" {
  description = "Compute Engine *machine* type that hosts the GPU. Must be compatible with gpu_type: N1 (e.g. n1-standard-4) for T4, G2 (e.g. g2-standard-8) for L4, A2 (e.g. a2-highgpu-1g) for A100."
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_count" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

variable "gpu_min_nodes" {
  description = "Minimum GPU nodes per zone in the autoscaling pool. Keep >=1 to hold a warm GPU (model cold-start is minutes)."
  type        = number
  default     = 1
}

variable "gpu_max_nodes" {
  description = "Maximum GPU nodes per zone the autoscaler may provision under load. Bounded by GPU quota and budget."
  type        = number
  default     = 3
}

variable "gpu_node_zones" {
  description = <<-EOT
    Zones the GPU node pool may place nodes in (all must be within var.region).
    Spanning every zone in the region + location_policy=ANY lets the autoscaler
    provision the GPU node in whichever zone currently has stock, instead of hard
    -failing on RESOURCE_POOL_EXHAUSTED in a single zone. NOTE: gpu_min/max_nodes
    are PER ZONE, so keep gpu_min_nodes=0 here or you hold one warm node *per zone*.
    Empty list = single-zone behavior (pool stays in var.zone).
  EOT
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# Fast LLM (LiteLLM -> Gemini)
# ---------------------------------------------------------------------------

# NOTE: the Gemini API key is NO LONGER a Terraform variable. It is read from GCP
# Secret Manager (secret id "gemini_api_key") in secrets.tf. Populate it out of
# band per the README before applying.

variable "llm_model" {
  description = "Gemini model the LiteLLM 'huri-fast' alias forwards to. Any fast, >4B model works."
  type        = string
  default     = "gemini-2.5-flash"
}

# ---------------------------------------------------------------------------
# Embedding server (llama.cpp -> bge-large-en-v1.5 GGUF)
# ---------------------------------------------------------------------------

variable "embedding_hf_repo" {
  description = "Hugging Face GGUF repo[:quant] llama.cpp pulls for the embedding server. MUST be the same bge-large-en-v1.5 quant used at ingestion, or query vectors won't match the stored ones."
  type        = string
  default     = "CompendiumLabs/bge-large-en-v1.5-gguf:Q4_K_M"
}

# ---------------------------------------------------------------------------
# Website image registry (the "GCR")
#
# HuRI owns the Artifact Registry that holds the website backend image, but is
# otherwise agnostic to the website itself. The website's own Terraform
# (HuRI_website_demo/deploy) builds the image reference from these same values
# and deploys it. Only the registry location/id are needed here.
# ---------------------------------------------------------------------------

variable "website_image_location" {
  description = "Artifact Registry location (region) for the website image repo, e.g. us-central1."
  type        = string
  default     = "us-central1"
}

variable "website_image_repo" {
  description = "Artifact Registry repository id holding the website image (imported, not created)."
  type        = string
  default     = "huri"
}
