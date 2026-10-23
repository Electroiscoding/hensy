#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
from pathlib import Path
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][hf_sync] %(message)s',
    handlers=[
        logging.FileHandler('/workspace/hensy/hf_upload.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('hf_sync')

CHECKPOINT_DIR = Path('/workspace/checkpoints/hensy_330m')
REPO_ID = 'Phase-Technologies/hensy-330m-pretrain'
STATE_FILE = Path('/workspace/checkpoints/.hf_uploaded_hensy.json')

def get_token():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if token:
        return token
    tpath = Path.home() / '.cache' / 'huggingface' / 'token'
    if tpath.exists():
        return tpath.read_text().strip()
    return ''

def load_uploaded():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return []
    return []

def save_uploaded(uploaded):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(uploaded, indent=2))

def sync_checkpoints(api: HfApi, token: str):
    if not CHECKPOINT_DIR.exists():
        return

    uploaded = load_uploaded()

    for ckpt in sorted(CHECKPOINT_DIR.iterdir()):
        if not ckpt.is_dir():
            continue
        tag = ckpt.name
        if tag in uploaded:
            continue

        model_file = ckpt / 'pytorch_model.bin'
        if not model_file.exists():
            continue

        s1 = model_file.stat().st_size
        time.sleep(3)
        s2 = model_file.stat().st_size
        if s1 != s2:
            logger.info(f'Checkpoint {tag} is still being written ({s1} -> {s2} bytes). Waiting.')
            continue

        logger.info(f'Starting upload for checkpoint {tag} to {REPO_ID} ({s2 / (1024*1024):.1f} MB)...')
        try:
            api.upload_folder(
                folder_path=str(ckpt),
                repo_id=REPO_ID,
                repo_type='model',
                path_in_repo=tag,
                token=token
            )
            logger.info(f'Successfully uploaded {tag} to Hugging Face: {REPO_ID}/{tag}')
            uploaded.append(tag)
            save_uploaded(uploaded)
        except Exception as e:
            logger.error(f'Upload failed for {tag}: {e}')

def main():
    token = get_token()
    if not token:
        logger.error('No HF token found! Checkpoint auto-sync cannot start.')
        return

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=REPO_ID, repo_type='model', exist_ok=True)
        logger.info(f'Verified repository exists: {REPO_ID}')
    except Exception as e:
        logger.warning(f'Repo check note: {e}')

    logger.info(f'Starting Hensy HF Auto-Sync Daemon -> {REPO_ID}')
    while True:
        try:
            sync_checkpoints(api, token)
        except Exception as e:
            logger.error(f'Sync loop error: {e}')
        time.sleep(10)

if __name__ == '__main__':
    main()
