"""Manual helper for uploading MLflow file store snapshot to Hugging Face Hub."""

from __future__ import annotations

import argparse

from optimizer_comparison.artifacts.hf_hub import get_hf_token, upload_mlflow_snapshot_to_hf


# /**
#  * Создает parser для загрузки zip snapshot-а MLflow file store в Hugging Face Hub.
#  *
#  * @return ArgumentParser для CLI-скрипта.
#  */
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload MLflow snapshot to Hugging Face Hub.")
    parser.add_argument("--mlruns-dir", default="outputs/mlruns")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-path", default="mlflow/mlruns_snapshot.zip")
    parser.add_argument("--snapshot-dir", default="outputs/mlflow_snapshots")
    parser.add_argument("--token-env-var", default="HF_TOKEN")
    return parser


# /**
#  * Запускает загрузку MLflow snapshot-а.
#  *
#  * @param argv CLI-аргументы или None для sys.argv.
#  * @return None.
#  */
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    token = get_hf_token(args.token_env_var)
    metadata = upload_mlflow_snapshot_to_hf(
        mlruns_dir=args.mlruns_dir,
        repo_id=args.repo_id,
        token=token,
        repo_path=args.repo_path,
        snapshot_dir=args.snapshot_dir,
    )
    print(metadata)


if __name__ == "__main__":
    main()
