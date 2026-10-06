# SciFact benchmark

Run the current checkout through the real MCP server: download SciFact, index its
5,183 documents with normal semantic chunking, execute 300 test queries and write
retrieval, context-size and latency measurements. No branch checkout is performed.

```bash
pnpm install
pnpm benchmark
```

The command builds the current source and passes `MODEL_NAME`, `CACHE_DIR`,
`RAG_DEVICE` and `RAG_DTYPE` directly to mcp-local-rag. Unset values use the server's
own defaults. Model files are downloaded as needed and reused from the model
cache. For example:

```bash
MODEL_NAME=onnx-community/embeddinggemma-300m-ONNX \
CACHE_DIR=/path/to/model-cache \
RAG_DEVICE=cpu \
pnpm benchmark --output benchmark/runs/gemma
```

Use `RAG_DEVICE=webgpu` for GPU execution, as with mcp-local-rag.
The benchmark fixes hybrid weight
at 0.6, returns 10 chunks, disables grouping/reranking/distance filters, uses a
50-character minimum chunk length and disables optional title/heading prefixes.
Automatic model-specific policies still apply. It uses an isolated database and
document directory; your existing `DB_PATH` and `BASE_DIR` are not used.

Downloads are cached in `benchmark/data/`. Each default invocation creates a new
run in `benchmark/runs/<timestamp>/`. Use `--output` to choose a retained directory.
Results include:

- `REPORT.md` and `summary.json`: aggregate quality, chunk sizes, tokens and latency.
- `per-query.json`: individual relevance metrics and rankings.
- `queries/*.json`: original MCP responses and measured durations.
- `index/` and `documents/`: the retained index and Markdown inputs.
- `manifest.json`, `sync-status.json`, `ingestion.json`, `server.log`: settings,
  code/dependency hashes, hardware and indexing progress.

To resume, run the same command with the same `--output` and model settings.
Unchanged documents and saved query results are reused. Changed code or settings
require a new directory. A completed run only recomputes its report from saved
results. If a process was forcibly terminated, ensure it has exited before
removing the run's `.lock` file and resuming.

nDCG and recall use BEIR's **document-level** relevance labels. Repeated documents
are collapsed at their first occurrence within the ten returned chunks. The
additional chunk-slot nDCG keeps original positions and assigns zero gain to
repeated documents. Latency includes embedding and MCP transport; one warm-up
per server session is excluded. Token counts include the complete MCP response,
but exclude agent prompts and follow-up calls. SciFact contains abstracts, so this
is not a benchmark of long papers, PDFs or generated answers.

No Python dependencies or separate evaluation service are needed. To check the
benchmark helpers without downloading models or running indexing:

```bash
node --test benchmark/benchmark.test.mjs
```

Sources: [BEIR](https://github.com/beir-cellar/beir),
[SciFact corpus schema](https://github.com/allenai/scifact/blob/master/doc/data.md#corpus).
