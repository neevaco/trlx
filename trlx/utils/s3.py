import os
import boto3
import logging
import threading
import tempfile

logger = logging.get_logger(__name__)

class _S3TransferCallback:
    def __init__(self, localfile: str) -> None:
        self._localfile = localfile
        self._total_transferred = 0
        self._last_logged = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_transferred: int) -> None:
        with self._lock:
            self._total_transferred += bytes_transferred
            if (self._total_transferred - self._last_logged) / 1024 // 1024 < 10:
                return
            if (self._total_transferred // (1024 * 1024)) % 50 == 0:
                logger.info(
                    f"Transferred {self._total_transferred/1024//1024} MB to S3 from {self._localfile}"
                )

def save_to_s3(local_path: str, remote_path: str) -> None:
    if not remote_path.startswith("s3://"):
        logger.error("Invalid s3 path")
    prefix_len = len("s3://")
    bucket, key = remote_path[prefix_len:].split("/", maxsplit=1)
    if key[-1] == "/":
        key = key[:-1]
    logger.info(f"Will upload model to s3://{bucket}/{key}...")
    s3 = boto3.resource("s3")

    def move_to_s3(localfile: str, remotekey: str) -> None:
        s3.Bucket(bucket).upload_file(
            localfile, remotekey, Callback=_S3TransferCallback(localfile)
        )

    if os.path.isfile(local_path):
        move_to_s3(local_path, key)
    else:
        for root, _, files in os.walk(local_path):
            for f in files:
                localfile = os.path.join(root, f)
                remotekey = f"{key}/{localfile[len(local_path)+1:]}"
                logger.info(f"Uploading {localfile} to s3://{bucket}/{remotekey}")
                move_to_s3(localfile, remotekey)
    logger.info("Finished copying to S3, goodbye")

def save_pretrained_s3(s3_path, model_or_tokenizer, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_or_tokenizer.save_pretrained(tmpdirname, **kwargs)
        save_to_s3(tmpdirname, s3_path)
