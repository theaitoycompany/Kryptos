"""Build an allowlisted static Space with a deterministic archive of the core."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".build" / "space"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "app.js", "worker.js"):
        shutil.copyfile(ROOT / "web" / name, OUT / name)
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(ROOT / name, OUT / name)
    archive = OUT / "kryptos.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted((ROOT / "src" / "kryptos").rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".txt"}:
                if path.is_symlink():
                    raise RuntimeError("Symlinks are not permitted in the Space build")
                info = zipfile.ZipInfo(str(path.relative_to(ROOT / "src")), (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, path.read_bytes(), compresslevel=9)
    manifest = {
        "archive": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "version": "0.1.1",
        "pyodide": "0.28.3",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        """---
title: Kryptos
emoji: 🔐
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
license: apache-2.0
short_description: Transcript de-identification that runs in your browser
---

# Kryptos

An open-source Python library for transcript de-identification.
This demo runs the dependency-free core locally in your browser using Pyodide.
The app does not upload your input, store it in browser storage, or call a
remote inference service. The browser downloads the runtime and application files.
Start with the included synthetic examples. Automated checks can miss identifiers.

Install locally, use optional model detectors, duplicate this Space, or reproduce
the published benchmarks from [the source repository](https://github.com/theaitoycompany/Kryptos).

This Space contains no model weights or production records. The optional GLiNER
model runs through the Python package; it is not loaded in this browser demo.
See the repository's benchmark report for measured limitations.
The [independent validation report](https://github.com/theaitoycompany/Kryptos/blob/main/docs/VALIDATION.md)
documents missed identifiers in some passed outputs. Human review is necessary;
this tool is not qualified for unattended publication of anonymized records.

The build command is `python scripts/build_space.py`. The archive hash is in
`manifest.json`; the worker checks it before loading the Python package.
The Pyodide runtime is pinned to 0.28.3 and retains its own license.
""",
        encoding="utf-8",
    )
    print(f"Built static Space ({archive.stat().st_size:,} byte core archive)")


if __name__ == "__main__":
    main()
