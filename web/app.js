const byId = (id) => document.getElementById(id);
const input = byId("input"),
  known = byId("known"),
  run = byId("run");
const output = byId("output"),
  verdict = byId("verdict"),
  feedback = byId("feedback");
const examples = {
  family: {
    text: "CHILD: My name is Aisha. I go to Greenfield School.\nPARENT: You can reach me at parent@example.com or call 07700 900461.",
    known: { CHILD_NAME: ["Aisha"], SCHOOL_NAME: ["Greenfield School"] },
  },
  asr: {
    text: "PARENT: my number is oh seven seven double oh nine double oh four six one and my email is alex dot example at gmail dot com",
    known: {},
  },
  context: {
    text: "CHILD: I'm nine and I'm the only goalkeeper on the under-10 Tigers team.\nPARENT: Dad owns the bakery opposite Westbrook Primary.",
    known: {},
  },
};
let worker,
  ready = false,
  busy = false,
  result = null,
  timeout,
  loadTimeout;

function setButton() {
  run.disabled = !ready || busy || !input.value.trim();
}
function resetResult() {
  result = null;
  output.textContent = "Your result will appear here.";
  verdict.textContent = "Ready to inspect";
  verdict.className = "badge";
  byId("summary").textContent = "Names, contact details, and contextual clues.";
  byId("copy").disabled = true;
  feedback.textContent = "";
  byId("count").textContent = `${input.value.length.toLocaleString()} / 6,000`;
  setButton();
}
function renderResult() {
  if (!result) return;
  const show =
    result.status === "passed" ||
    (result.status === "review" && byId("allow-review").checked);
  output.textContent = show
    ? result.text
    : result.status === "blocked"
      ? "Text withheld. The checks found possible remaining identifiers. Add known identifiers or inspect this input with the local Python library."
      : "Text withheld for review. Enable ‘Show text marked for review’ to inspect the candidate locally.";
  verdict.textContent = {
    passed: "Checks passed",
    review: "Review required",
    blocked: "Blocked",
  }[result.status];
  verdict.className = `badge ${result.status}`;
  byId("summary").textContent =
    `${result.spans} detected spans · ${result.findings.length} review findings`;
  byId("copy").disabled = !show || !result.text;
  feedback.textContent = result.findings.length
    ? result.findings.join(" · ")
    : "Automated checks passed. Inspect the result before sharing.";
}
function startWorker() {
  clearTimeout(loadTimeout);
  if (worker) worker.terminate();
  ready = false;
  byId("engine").textContent = "Loading local engine…";
  setButton();
  worker = new Worker("worker.js");
  const activeWorker = worker;
  loadTimeout = setTimeout(() => {
    if (worker !== activeWorker) return;
    worker.terminate();
    ready = false;
    busy = false;
    setButton();
    byId("engine").textContent = "Engine could not load";
    feedback.textContent =
      "The engine download timed out. Check your connection and reload.";
  }, 90000);
  worker.onmessage = ({ data }) => {
    if (worker !== activeWorker) return;
    if (data.type === "ready") {
      clearTimeout(loadTimeout);
      ready = true;
      byId("engine").textContent = "● Local engine ready";
      setButton();
      return;
    }
    clearTimeout(timeout);
    busy = false;
    setButton();
    if (data.type === "result") {
      result = data.result;
      renderResult();
    } else if (data.type === "load-error") {
      clearTimeout(loadTimeout);
      ready = false;
      setButton();
      byId("engine").textContent = "Engine could not load";
      feedback.textContent =
        "Check your connection and reload to download the local engine. Your text has not been uploaded.";
    } else {
      verdict.textContent = "Input could not be processed";
      feedback.textContent =
        "Could not process this input. Check the known-identifier JSON and try a shorter transcript.";
    }
  };
  worker.onerror = () => {
    if (worker !== activeWorker) return;
    clearTimeout(timeout);
    clearTimeout(loadTimeout);
    busy = false;
    ready = false;
    setButton();
    byId("engine").textContent = "Engine unavailable";
    feedback.textContent = "Reload this page to restart the local engine.";
  };
  worker.postMessage({ type: "initialize" });
}
function chooseExample() {
  const example = examples[byId("example").value];
  input.value = example.text;
  known.value = JSON.stringify(example.known, null, 2);
  resetResult();
}
byId("form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (run.disabled) return;
  let identifiers;
  try {
    identifiers = JSON.parse(known.value.trim() || "{}");
    if (
      !identifiers ||
      Array.isArray(identifiers) ||
      typeof identifiers !== "object"
    )
      throw new Error();
  } catch {
    feedback.textContent =
      'Known identifiers must be a JSON object, such as {"CHILD_NAME": ["Aisha"]}.';
    return;
  }
  resetResult();
  busy = true;
  setButton();
  verdict.textContent = "Processing locally…";
  worker.postMessage({
    type: "process",
    text: input.value,
    known: identifiers,
  });
  timeout = setTimeout(() => {
    busy = false;
    startWorker();
    verdict.textContent = "Processing stopped";
    feedback.textContent =
      "Processing took too long. Try a shorter transcript when the engine is ready.";
  }, 45000);
});
byId("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(output.textContent);
    feedback.textContent = "Copied to clipboard.";
  } catch {
    feedback.textContent =
      "Clipboard unavailable. Select and copy the result manually.";
  }
});
byId("clear").addEventListener("click", () => {
  clearTimeout(timeout);
  input.value = "";
  known.value = "{}";
  busy = false;
  resetResult();
  startWorker();
  input.focus();
});
function onEdit() {
  if (busy) {
    clearTimeout(timeout);
    busy = false;
    startWorker();
  }
  resetResult();
}
input.addEventListener("input", onEdit);
known.addEventListener("input", onEdit);
byId("example").addEventListener("change", () => {
  if (busy) {
    clearTimeout(timeout);
    busy = false;
    startWorker();
  }
  chooseExample();
});
byId("allow-review").addEventListener("change", renderResult);
chooseExample();
startWorker();
