"""Reject unexpected files and credential patterns before publication.

This is an additional mechanical check, not proof that code or data is safe
to publish. The release uses an explicit inventory and human-readable diff.
Only filenames and rule names are printed on failure; matched bytes are not.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "MANIFEST.in",
    "pyproject.toml",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "playwright.config.js",
}
DIRECTORIES = {
    "src": {".py", ".json", ".txt"},
    "tests": {".py", ".js"},
    "benchmarks": {".py"},
    "scripts": {".py"},
    "docs": {".md", ".json"},
    "examples": {".txt", ".json"},
    "web": {".html", ".css", ".js"},
    ".github": {".yml"},
}
PATTERNS = {
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{25,}"),
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "jwt": re.compile(rb"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),
    "local_user_path": re.compile(rb"/(?:Users|home)/[A-Za-z][A-Za-z0-9._-]+/"),
    "database_endpoint": re.compile(rb"[a-z]{20}\.supabase\.co"),
}


def inventory():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(set(path for path in result.stdout.decode().split("\0") if path))


def main():
    failures, manifest = [], {}
    for name in inventory():
        path = ROOT / name
        relative = Path(name)
        allowed = name in ROOT_FILES or (
            len(relative.parts) > 1 and relative.suffix in DIRECTORIES.get(relative.parts[0], set())
        )
        if not allowed or path.is_symlink() or not path.is_file():
            failures.append((name, "unexpected_file"))
            continue
        content = path.read_bytes()
        for rule, pattern in PATTERNS.items():
            if pattern.search(content):
                failures.append((name, rule))
        manifest[name] = hashlib.sha256(content).hexdigest()
    if failures:
        for name, rule in failures:
            print(f"Rejected {name}: {rule}")
        raise SystemExit(1)
    (ROOT / ".build").mkdir(exist_ok=True)
    (ROOT / ".build" / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Publication inventory passed: {len(manifest)} allowlisted files")


if __name__ == "__main__":
    main()
