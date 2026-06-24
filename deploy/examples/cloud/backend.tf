# Remote Terraform state in GCS (versioned, IAM-restricted). Holds the cluster
# state — and, because Terraform decrypts Secret Manager values at apply time,
# secret material too. Keep the bucket private.
#
# The bucket must EXIST before `terraform init` (chicken-and-egg: backends can't
# create their own bucket and can't interpolate variables). It is supplied at
# init time, so only the static prefix is committed here:
#
#   terraform init -backend-config="bucket=<your-tf-state-bucket>"
#
# See gcp_steps.md §0 for creating the bucket.
terraform {
  backend "gcs" {
    prefix = "huri-cloud"
  }
}
