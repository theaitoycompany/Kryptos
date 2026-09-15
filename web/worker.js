/* The worker receives text locally. Network requests only load static assets. */
const runtimeURL = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
let python;

async function initialize() {
  importScripts(`${runtimeURL}pyodide.js`);
  python = await loadPyodide({
    indexURL: runtimeURL,
    stdout: () => {},
    stderr: () => {},
  });
  const manifestResponse = await fetch("manifest.json");
  if (!manifestResponse.ok) throw new Error("Manifest unavailable");
  const manifest = await manifestResponse.json();
  const archiveResponse = await fetch("kryptos.zip");
  if (!archiveResponse.ok) throw new Error("Package unavailable");
  const archive = await archiveResponse.arrayBuffer();
  const hash = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", archive)),
  )
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  if (hash !== manifest.sha256)
    throw new Error("Package integrity check failed");
  python.unpackArchive(archive, "zip", { extractDir: "/home/pyodide" });
  python.runPython(`
import json
from kryptos import Config, Pipeline

def process_request(raw):
    payload = json.loads(raw)
    text, known = payload["text"], payload["known"]
    if len(text) > 6000 or not isinstance(known, dict) or len(known) > 54:
        raise ValueError("Invalid input")
    config = Config.load()
    for entity, values in known.items():
        if entity not in config.taxonomy.entities or not isinstance(values, list):
            raise ValueError("Invalid known identifiers")
        if len(values) > 30 or any(not isinstance(v, str) or len(v) > 256 for v in values):
            raise ValueError("Invalid known identifiers")
    result = Pipeline(config).process_text(text, known_values=known)
    return json.dumps({
        "status": result.status,
        "text": result.sanitized_text if result.status != "blocked" else "",
        "spans": len(result.spans),
        "findings": [f.message for f in result.qa],
    })
`);
  postMessage({ type: "ready" });
}

onmessage = async ({ data }) => {
  if (data.type === "initialize") {
    try {
      await initialize();
    } catch {
      postMessage({ type: "load-error" });
    }
    return;
  }
  try {
    python.globals.set("request_json", JSON.stringify(data));
    const result = JSON.parse(
      python.runPython("process_request(request_json)"),
    );
    postMessage({ type: "result", result });
  } catch {
    postMessage({ type: "process-error" });
  } finally {
    if (python) python.globals.delete("request_json");
  }
};
