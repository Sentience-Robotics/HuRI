# Secrets come from GCP Secret Manager, not from variables or env vars. Create
# the secret and add a version OUT OF BAND before applying (see README §1);
# Terraform only *reads* the latest version here, so the plaintext key is never
# a Terraform input and never touches git.
#
# Caveat: the decrypted value still lands in Terraform state, so keep state in a
# secured remote backend (GCS). The identity running Terraform needs
# roles/secretmanager.secretAccessor on this secret.
data "google_secret_manager_secret_version" "gemini_api_key" {
  secret = "gemini_api_key"
  # version defaults to "latest".
}
