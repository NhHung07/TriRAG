#!/usr/bin/env python3
"""Xây dựng chỉ mục (index) cho một hoặc nhiều phương pháp truy hồi."""

from __future__ import annotations

import argparse
import logging

from trirag.config import load_config
from trirag.eval.runner import ExperimentRunner
from trirag.utils import load_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Đường dẫn tới file YAML cấu hình")
    ap.add_argument("--methods", default="", help="Danh sách phương pháp cách nhau bởi dấu phẩy (mặc định: tất cả)")
    ap.add_argument("--rebuild", action="store_true", help="Xoá và dựng lại chỉ mục từ đầu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runner = ExperimentRunner(cfg)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(cfg["methods"].keys())
    for m in methods:
        runner.build(m, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
