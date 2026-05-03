"""Manual helper for downloading and merging MLflow file store snapshot."""

from __future__ import annotations

import argparse

from optimizer_comparison.artifacts.hf_hub import (
    download_file_from_hf,
    get_hf_token,
    merge_mlflow_snapshot_archive,
)


# /**
#  * Создает parser для скачивания MLflow snapshot-а из Hugging Face Hub.
#  *
#  * @return ArgumentParser для CLI-скрипта.
#  */
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and merge MLflow snapshot.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-path", default="mlflow/mlruns_after_remote_train.zip")
    parser.add_argument("--mlruns-dir", default="outputs/mlruns")
    parser.add_argument("--download-dir", default="outputs/mlflow_downloads")
    parser.add_argument("--extract-dir", default="outputs/mlflow_downloads/extracted")
    parser.add_argument("--token-env-var", default="HF_TOKEN")
    parser.add_argument("--revision", default=None)
    return parser


# /**
#  * Скачивает zip snapshot MLflow и сливает его с локальным outputs/mlruns.
#  *
#  * @param argv CLI-аргументы или None для sys.argv.
#  * @return None.
#  */
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    token = get_hf_token(args.token_env_var)
    archive_path = download_file_from_hf(
        repo_id=args.repo_id,
        target_dir=args.download_dir,
        repo_path=args.repo_path,
        token=token,
        revision=args.revision,
    )
    merged_path = merge_mlflow_snapshot_archive(
        archive_path=archive_path,
        target_mlruns_dir=args.mlruns_dir,
        extract_dir=args.extract_dir,
    )
    print({"archive_path": str(archive_path), "merged_path": str(merged_path)})


if __name__ == "__main__":
    main()
