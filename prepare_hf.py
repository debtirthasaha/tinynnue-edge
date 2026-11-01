"""Stage the Hugging Face model and dataset repos under hf/, optionally upload.

    python prepare_hf.py                      # stage only, nothing leaves the machine
    python prepare_hf.py --upload --token-file <path to write token>
"""
import argparse
import glob
import os
import shutil

MODEL_REPO = "MR0b0t/tinynnue-edge"
DATASET_REPO = "MR0b0t/tinynnue-stockfish-labels"


def stage():
    shutil.rmtree("hf", ignore_errors=True)
    os.makedirs("hf/model/results")
    os.makedirs("hf/dataset")
    for src in ("checkpoints/model.safetensors", "export/tinynnue_int8.bin", "export/tinynnue_fp32.bin"):
        shutil.copy(src, "hf/model/")
    shutil.copy("hf_cards/model_card.md", "hf/model/README.md")
    for src in glob.glob("results/*.json") + ["results/RESULTS.md"]:
        if "smoke" not in src:
            shutil.copy(src, "hf/model/results/")
    shutil.copy("data/positions.csv", "hf/dataset/positions.csv")
    shutil.copy("hf_cards/dataset_card.md", "hf/dataset/README.md")
    for d in ("hf/model", "hf/dataset"):
        print(d, sorted(os.listdir(d)))


def upload(token_file):
    from huggingface_hub import HfApi
    api = HfApi(token=open(token_file).read().strip())
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path="hf/model", repo_id=MODEL_REPO, repo_type="model")
    api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path="hf/dataset", repo_id=DATASET_REPO, repo_type="dataset")
    print(f"https://huggingface.co/{MODEL_REPO}\nhttps://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--token-file")
    args = ap.parse_args()
    stage()
    if args.upload:
        upload(args.token_file)
