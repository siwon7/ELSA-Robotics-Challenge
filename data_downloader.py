import argparse
import gzip
import json
import os
import time
from pathlib import Path
from urllib import request

NUM_PROCESSES = 64
DV_URL = "https://dataverse.harvard.edu/"
API_TOKEN = os.environ.get("ELSA_DATAVERSE_API_TOKEN", "")
DATA_ROOT = Path(os.environ.get("ELSA_DATA_ROOT", "./datasets"))
USER_AGENT = os.environ.get(
    "ELSA_DOWNLOAD_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
)
DOWNLOAD_RETRIES = int(os.environ.get("ELSA_DOWNLOAD_RETRIES", "5"))
DOWNLOAD_RETRY_SLEEP = float(os.environ.get("ELSA_DOWNLOAD_RETRY_SLEEP_SECS", "5"))


def is_valid_download(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    if path.suffix == ".gz":
        try:
            with gzip.open(path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
        except OSError:
            return False

    return True


def download_file(file_id: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_download(destination):
        return

    part_path = destination.with_suffix(destination.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    last_error = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            request.urlretrieve(f"{DV_URL}api/access/datafile/{file_id}", str(part_path))
            if not is_valid_download(part_path):
                raise OSError(f"Downloaded file is invalid: {part_path}")
            part_path.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            if part_path.exists():
                part_path.unlink()
            if attempt == DOWNLOAD_RETRIES:
                break
            print(f"Retrying {destination.name} ({attempt}/{DOWNLOAD_RETRIES}) after: {exc}")
            time.sleep(DOWNLOAD_RETRY_SLEEP)

    raise last_error


def download_envs(path, task, envs, files):
    for env in envs:
        file_ids = [file for file in files if f"{path}/{env}" in file["directoryLabel"]]
        print(f"Downloading {env}")
        env_root = DATA_ROOT / path / env
        env_root.mkdir(parents=True, exist_ok=True)
        for file_id in file_ids:
            if "episodes_observations.pkl.gz" in file_id["dataFile"]["filename"]:
                obs_id = file_id["dataFile"]["id"]
                download_file(obs_id, env_root / "episodes_observations.pkl.gz")
            elif "variation_descriptions.pkl" in file_id["dataFile"]["filename"]:
                var_id = file_id["dataFile"]["id"]
                download_file(var_id, env_root / "variation_descriptions.pkl")

    task_root = DATA_ROOT / path
    if not (task_root / f"{task}_fed.json").exists():
        file_id = [file for file in files if f"{task}_fed.json" in file["label"]][0]["dataFile"]["id"]
        download_file(file_id, task_root / f"{task}_fed.json")

    if not (task_root / f"{task}_fed.yaml").exists():
        file_id = [file for file in files if f"{task}_fed.yaml" in file["label"]][0]["dataFile"]["id"]
        download_file(file_id, task_root / f"{task}_fed.yaml")


def download(envs_per_chunk, task, path, pid, data_type, num_envs):
    print(f"Downloading to {(DATA_ROOT / path).resolve()}")

    start_env = 0
    if data_type == "eval":
        start_env = 400

    if data_type == "test":
        len_dataset = 1
    elif num_envs == 400:
        if data_type == "training":
            len_dataset = 400
        elif data_type == "eval":
            len_dataset = 50
    else:
        if data_type == "training" and num_envs <= 400:
            len_dataset = num_envs
        elif data_type == "eval" and num_envs <= 50:
            len_dataset = num_envs
        else:
            raise ValueError("Number of environments should be maximum 400 for training and 50 for eval")

    envs = ["env_" + str(i) for i in range(start_env, len_dataset + start_env)]
    envs = sorted(envs, key=lambda x: int(x.split("_")[-1]))
    chunk_envs = [envs[i : i + envs_per_chunk] for i in range(0, len(envs), envs_per_chunk)]

    files = json.loads(
        request.urlopen(f"{DV_URL}api/datasets/:persistentId?persistentId={pid}").read().decode("utf-8")
    )["data"]["latestVersion"]["files"]
    files = [file for file in files if task in file["directoryLabel"]]

    for i, envs in enumerate(chunk_envs):
        print(f"\n --> Downloading envs {i * envs_per_chunk} to {(i + 1) * envs_per_chunk}")
        download_envs(path, task, envs, files)


def install_API_token():
    opener = request.build_opener()
    opener.addheaders = [("User-Agent", USER_AGENT)]
    if API_TOKEN:
        opener.addheaders.append(("X-Dataverse-key", API_TOKEN))
    request.install_opener(opener)


def main(args):
    envs_per_chunk = 5

    if args.data_type == "eval":
        pid = "doi:10.7910/DVN/DOZY6N"
    elif args.data_type == "test":
        pid = "doi:10.7910/DVN/OVVN2E"
    elif args.data_type == "training":
        if args.task == "slide_block_to_target":
            pid = "doi:10.7910/DVN/XEAYPQ"
        elif args.task == "close_box":
            pid = "doi:10.7910/DVN/QPUOJH"
        elif args.task == "insert_onto_square_peg":
            pid = "doi:10.7910/DVN/PSPXJK"
        elif args.task == "scoop_with_spatula":
            pid = "doi:10.7910/DVN/EPU7UW"

    path = f"{args.data_type}/{args.task}"

    install_API_token()
    download(envs_per_chunk, args.task, path, pid, args.data_type, args.num_envs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="training", choices=["training", "eval", "test"])
    parser.add_argument(
        "--task",
        type=str,
        default="close_box",
        choices=["slide_block_to_target", "close_box", "insert_onto_square_peg", "scoop_with_spatula"],
    )
    parser.add_argument("--num_envs", type=int, default=400)
    args = parser.parse_args()
    main(args)
