# Optional model detectors

Models execute on your machine. Kryptos does not send transcripts to the Hugging
Face Inference API. The public browser demo runs the core detectors only.

## Benchmarked GLiNER profile

| Property | Value |
|---|---|
| Model | [`urchade/gliner_multi_pii-v1`](https://huggingface.co/urchade/gliner_multi_pii-v1) |
| Revision | `1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d` |
| Tokenizer/backbone configuration | `microsoft/mdeberta-v3-base` at `a0484667b22365f84929a935b5e50a51f71f159d` |
| Weight license | Apache-2.0, per upstream model card |
| Adapter | `gliner_child` |
| Threshold | 0.28 |
| Device used for published diagnostics | CPU, four PyTorch threads |
| Python packages | `gliner==0.2.24`, `transformers==5.16.1`, `torch==2.14.0` |

```bash
uv sync --locked --extra ml --extra fast
uv run --locked --extra ml --extra fast kryptos run examples/transcript.txt \
  --known examples/known.json --enable gliner_child
```

The configured child/family schema includes names, schools, addresses, relatives,
contact details, and contextual identifiers. Long inputs use overlapping bounded
windows. Model availability, finite outputs, span bounds, and complete prediction
batches are checked. These checks do not prove semantic coverage.

After the first successful download, `HF_HUB_OFFLINE=1` prevents Hub requests.
Set `HF_HOME` to choose the local model cache. Do not put credentials in a repository.
Public model downloads work without a token.

## Generic Hugging Face token classifier

The `hf_token` adapter supports fast tokenizers with offsets and token-label logits.
It decodes BIO/BIOES labels and overlapping windows. `opf` is an opt-in configuration
for [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter), pinned to
`7ffa9a043d54d1be65afb281eddf0ffbe629385b` (Apache-2.0 per upstream).

This release does not benchmark the full OPF model. Its adapter and configuration
are provided for local evaluation; architecture, memory, and device compatibility
must be checked before relying on it. The default and GLiNER profiles do not load it.

To use your own compatible detector, supply a detector spec with `type`, `name`,
`model`, an immutable `revision`, and the correct taxonomy label map. Model APIs are
not interchangeable: a GLiNER2 model is not automatically compatible with GLiNER.
The token-classifier adapter disables remote model code unless explicitly enabled.
GLiNER uses its upstream loader; use trusted checkpoints. The benchmarked configuration
pins the model and the separate tokenizer/backbone assets and requires strict weight loading.

## Custom weights

Kryptos ships no custom-trained model. The published experiments evaluate existing
weights and synthetic regression fixtures. Training a useful replacement needs a
separate reviewed training corpus and independent evaluation; these benchmark
templates should not be used to claim generalization after training on them.

Upstream model cards govern weight licensing. The repository license covers Kryptos
code; it does not replace model or dependency licenses.
