from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

# hf_hub_download(repo_id="mmaaz60/LLaVA-7B-Lightening-v1-1", filename="config.json")
snapshot_download(repo_id="mmaaz60/LLaVA-7B-Lightening-v1-1", local_dir="LLaVA-7B-Lightening-v1-1")
# hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")