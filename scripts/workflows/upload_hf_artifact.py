"""Manual helper for uploading one local artifact to Hugging Face Hub."""

from __future__ import annotations

import argparse

from optimizer_comparison.artifacts.hf_hub import get_hf_token, upload_path_to_hf


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI parser for uploading one artifact to Hugging Face Hub."""
    parser = argparse.ArgumentParser(description="Upload one artifact to Hugging Face Hub.")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--token-env-var", default="HF_TOKEN")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Uploads one local artifact to Hugging Face Hub."""
    args = build_parser().parse_args(argv)
    token = get_hf_token(args.token_env_var)
    metadata = upload_path_to_hf(
        artifact_path=args.artifact_path,
        repo_id=args.repo_id,
        repo_path=args.repo_path,
        token=token,
        commit_message="Manual artifact upload",
    )
    print(metadata)


if __name__ == "__main__":
    main()
