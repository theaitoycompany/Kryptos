# Security

Do not submit real transcripts, personal identifiers, credentials, or raw identity
mappings in issues, pull requests, benchmark fixtures, or Space feedback. Use a minimal
synthetic reproduction. Report sensitive vulnerabilities through GitHub's private
vulnerability reporting feature when available; public issues should contain no secrets.

Kryptos performs heuristic de-identification. A passing status does not prove anonymity,
regulatory compliance, or safe publication. Review candidate text and combinations of
contextual details. Blocked API results can contain residual identifiers; the CLI and
demo withhold them. Protect source files, Python objects, outputs, and logs appropriately.

The core has no network transport. Optional models download external files and execute
locally. The token-classifier adapter disables remote model code by default. Use trusted
checkpoints with the upstream GLiNER loader. The static demo downloads a pinned
Pyodide runtime and application assets; processing does not upload text or use browser
storage. Clearing the demo terminates its worker. No secure memory-erasure guarantee is made.

The publication check uses an allowlist and credential patterns. It supplements source
review; it cannot prove that arbitrary content is free of PII. No production data,
identity references, model caches, or credentials belong in this repository.
