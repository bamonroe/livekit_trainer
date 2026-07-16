#!/usr/bin/env python3
"""Receive Android training bundle uploads and import them into data/real."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class SyncServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], repo_root: Path) -> None:
        super().__init__(address, SyncHandler)
        self.repo_root = repo_root


class SyncHandler(BaseHTTPRequestHandler):
    server: SyncServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self.respond_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self.respond_json(HTTPStatus.OK, {"status": "ok"})

    def do_POST(self) -> None:
        if self.path != "/sync":
            self.respond_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        content_length = self.headers.get("Content-Length")
        if not content_length:
            self.respond_json(HTTPStatus.LENGTH_REQUIRED, {"error": "missing Content-Length"})
            return

        try:
            length = int(content_length)
        except ValueError:
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
            return

        if length <= 0:
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "empty upload"})
            return
        if length > 512 * 1024 * 1024:
            self.respond_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "upload too large"})
            return

        incoming_dir = self.server.repo_root / "incoming" / "bundles"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        archive = incoming_dir / f"bundle_{int(time.time() * 1000)}.zip"
        archive.write_bytes(self.rfile.read(length))

        try:
            result = self.import_archive(archive)
        except Exception as error:  # noqa: BLE001 - return a useful HTTP error body.
            self.respond_json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(error), "archive": str(archive.relative_to(self.server.repo_root))},
            )
            return

        self.respond_json(HTTPStatus.OK, result)

    def import_archive(self, archive: Path) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="livekit_bundle_") as tmp:
            bundle_dir = Path(tmp) / "bundle"
            bundle_dir.mkdir()
            safe_extract_zip(archive, bundle_dir)

            validate = run_script(self.server.repo_root, "validate_bundle.py", bundle_dir)
            imported = run_script(
                self.server.repo_root,
                "import_bundle.py",
                bundle_dir,
                "--skip-existing",
            )
            manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))

        return {
            "status": "imported",
            "archive": str(archive.relative_to(self.server.repo_root)),
            "wake_word_slug": manifest["wake_word"]["slug"],
            "clip_count": len(manifest["clips"]),
            "validate_output": validate,
            "import_output": imported,
        }

    def respond_json(self, status: HTTPStatus, body: dict[str, object]) -> None:
        payload = json.dumps(body, sort_keys=True).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}", file=sys.stderr)


def run_script(repo_root: Path, script: str, *args: object) -> str:
    command = [sys.executable, str(repo_root / "scripts" / script), *(str(arg) for arg in args)]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed: {output}")
    return output


def safe_extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (dest / member.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise RuntimeError(f"zip member escapes bundle: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Bind host, default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=8765, help="Bind port, default: 8765")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root, default: parent of scripts/",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    server = SyncServer((args.host, args.port), repo_root)
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"Importing bundles into {repo_root / 'data' / 'real'}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
