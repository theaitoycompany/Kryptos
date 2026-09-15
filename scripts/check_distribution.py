"""Install the built wheel offline in a clean environment outside the checkout."""

import os
import subprocess
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    wheels = sorted((ROOT / "dist").glob("kryptos_pii-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Build exactly one wheel in dist/ before running this check")
    with tempfile.TemporaryDirectory(prefix="kryptos-install-") as directory:
        work = Path(directory)
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(work / "venv")
        python = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])],
            cwd=work,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                """
from kryptos import Pipeline, Config
config = Config.load()
assert len(config.taxonomy.entities) == 54
pipeline = Pipeline(config)
context = next(d for d in pipeline.detectors if d.name == 'context')
assert 'aisha' in context.first_names
result = pipeline.process_text('My name is Aisha. Email parent@example.com.',
                               known_values={'CHILD_NAME': ['Aisha']})
assert 'Aisha' not in result.sanitized_text
assert 'parent@example.com' not in result.sanitized_text
assert result.status == 'passed'
print('Installed-wheel API and package resources passed')
""",
            ],
            cwd=work,
            check=True,
        )
        subprocess.run([str(python), "-I", "-m", "kryptos", "selftest"], cwd=work, check=True)
    print("Clean, offline wheel installation passed")


if __name__ == "__main__":
    main()
