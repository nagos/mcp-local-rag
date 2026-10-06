import fs from 'node:fs/promises'
import path from 'node:path'

export async function writeReport(output, manifest, summary) {
  const config = manifest.identity.config
  const lines = [
    '# SciFact MCP benchmark',
    '',
    `- Current checkout: \`${manifest.commit ?? 'unavailable'}\`; built-code SHA-256: \`${manifest.identity.code}\`.`,
    `- Model: \`${config.modelName}\`; device: \`${config.device}\`; dtype: \`${config.dtype ?? 'server default'}\`.`,
    `- ${summary.documents} documents; ${summary.queries} test queries; hybrid weight 0.6; top 10 chunks.`,
    '- Normal MCP bulk sync and semantic chunking; no grouping, reranking or distance filter.',
    '',
    '| Metric | Result |',
    '|---|---:|',
    `| Document nDCG@10 | ${summary.ndcgAt10.toFixed(4)} |`,
    `| Chunk-slot nDCG@10 | ${summary.chunkSlotNdcgAt10.toFixed(4)} |`,
    `| Recall@10 | ${summary.recallAt10.toFixed(4)} |`,
    `| MRR@10 (chunk slots) | ${summary.chunkSlotMrrAt10.toFixed(4)} |`,
    `| Relevant document at rank 1 | ${(100 * summary.relevantAt1).toFixed(1)}% |`,
    `| Chunks | ${summary.chunks} |`,
    `| Median / p95 chunk tokens | ${summary.chunkTokens.median} / ${summary.chunkTokens.p95} |`,
    `| Mean returned chunk-text tokens | ${summary.chunkTextTokens.toFixed(0)} |`,
    `| Mean complete MCP response tokens | ${summary.mcpTextTokens.toFixed(0)} |`,
    `| Mean distinct documents in top 10 | ${summary.uniqueDocuments.toFixed(2)} |`,
    `| Median / p95 MCP query latency, ms | ${summary.queryMilliseconds.median.toFixed(1)} / ${summary.queryMilliseconds.p95.toFixed(1)} |`,
    `| Indexing seconds (last completed sync) | ${summary.ingestionSeconds.toFixed(1)} |`,
    '',
    'Document nDCG collapses repeated documents at their first occurrence within the ten returned chunks. Chunk-slot nDCG retains the original positions and gives duplicate documents zero gain. Both use linear relevance gain, as in BEIR/trec_eval. No extra chunks are fetched to fill the document ranking.',
    '',
    'Token counts use the model tokenizer without truncation. Chunk-text counts include special tokens per chunk; complete-response counts include JSON and metadata. These exclude agent prompts and follow-up calls.',
    '',
    'Latency covers query embedding, MCP execution and transport, after one warm-up per server session. Resumed runs can contain measurements from different sessions. Indexing time is the last completed sync, and excludes earlier interrupted work.',
    '',
    'SciFact contains scientific abstracts and document-level relevance labels. These metrics do not establish evidence quality inside individual chunks or behavior on long PDFs.',
    '',
    'The index, documents, raw MCP responses and per-query measurements are retained in this directory. The manifest records settings, source/dependency hashes and hardware. A matching output directory resumes unfinished work; changed code or settings require a new directory.',
    '',
    'Dataset: [BEIR](https://github.com/beir-cellar/beir); corpus schema: [SciFact](https://github.com/allenai/scifact/blob/master/doc/data.md#corpus).',
  ]
  await fs.writeFile(path.join(output, 'REPORT.md'), `${lines.join('\n')}\n`)
}
