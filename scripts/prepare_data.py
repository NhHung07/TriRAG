#!/usr/bin/env python3
"""Tiền xử lý dữ liệu theo file cấu hình YAML (hỗ trợ tự tải HotpotQA / QASPER / UltraDomain)."""

from __future__ import annotations

import argparse
import logging
import tarfile
import urllib.request
from pathlib import Path

from trirag.config import load_config
from trirag.data.io import build_dataset, processed_dir, save_dataset
from trirag.retrievers import Components
from trirag.utils import PROJECT_ROOT, load_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_data")

DOWNLOAD_URLS: dict[str, list[str]] = {
    "hotpotqa": [
        "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    ],
    "qasper": [
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    ],
    "ultradomain": [
        "https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/cs.jsonl",
    ],
}


def _download_with_mirrors(urls: list[str], dest: Path) -> None:
    last_err = None
    for url in urls:
        try:
            log.info("Đang tải %s -> %s ...", url, dest)
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            log.warning("Không tải được từ %s (%s), thử mirror tiếp theo...", url, e)
            last_err = e
    raise RuntimeError(f"Không thể tải {dest} từ các URL: {urls}") from last_err


def ensure_raw_data(cfg: dict) -> None:
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    if name in ("erqa", "vi_docs"):
        return
    raw_rel = ds_cfg.get("path", "")
    target = Path(raw_rel) if Path(raw_rel).is_absolute() else PROJECT_ROOT / raw_rel
    if target.exists():
        return
    urls = DOWNLOAD_URLS.get(name)
    if not urls:
        raise FileNotFoundError(f"Chưa có file dữ liệu {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if name == "qasper" and urls[0].endswith(".tgz"):
        tgz_path = target.parent / "qasper-train-dev-v0.3.tgz"
        _download_with_mirrors(urls, tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(target.parent)
    else:
        _download_with_mirrors(urls, target)
    log.info("Đã tải xong: %s (%.1f MB)", target, target.stat().st_size / 1e6)


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Đường dẫn tới file YAML cấu hình")
    ap.add_argument("--no-download", action="store_true", help="Không tự tải dữ liệu thô nếu thiếu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not args.no_download:
        ensure_raw_data(cfg)

    llm = None
    if cfg["dataset"]["name"] == "ultradomain":
        llm = Components(cfg).index_llm

    ds = build_dataset(cfg, llm=llm)
    out = processed_dir(cfg)
    save_dataset(ds, out)
    log.info("Đã lưu dữ liệu tiền xử lý tại %s: %s", out, ds.stats())


if __name__ == "__main__":
    main()
