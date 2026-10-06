import { appendFileSync } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import { performance } from 'node:perf_hooks'
import { connect } from '@lancedb/lancedb'
import { AutoTokenizer, env } from '@huggingface/transformers'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { readJson, saveJson } from './dataset.mjs'
import { distribution, evaluate, mean, parseToolResult, rankHits } from './metrics.mjs'

const limit = 10
const timeout = 180_000

export async function runMcp(output, environment, dataset, root) {
  const client = new Client({ name: 'scifact-benchmark', version: '1.0.0' })
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [path.join(root, 'dist/index.js')],
    cwd: root,
    env: environment,
    stderr: 'pipe',
  })
  transport.stderr?.on('data', (chunk) => appendFileSync(path.join(output, 'server.log'), chunk))
  const call = async (name, args) =>
    client.callTool({ name, arguments: args }, undefined, { timeout })
  try {
    await client.connect(transport)
    await saveJson(path.join(output, 'server-info.json'), { version: client.getServerVersion() })
    if (!(await readJson(path.join(output, 'ingestion.json')))) {
      const started = performance.now()
      const { jobId } = parseToolResult(await call('sync_start', { path: environment.BASE_DIR }))
      let progress = -1
      for (;;) {
        const job = parseToolResult(await call('sync_status', { jobId }))
        await saveJson(path.join(output, 'sync-status.json'), job)
        if (Math.floor(job.completed / 100) !== progress) {
          progress = Math.floor(job.completed / 100)
          console.log(`Indexing: ${job.completed}/${job.total}`)
        }
        if (job.state === 'succeeded') {
          if (
            job.total !== dataset.corpus.length ||
            job.summary.pruned ||
            job.warnings.length ||
            job.summary.empty
          ) {
            throw new Error(`Incomplete corpus ingestion: ${JSON.stringify(job)}`)
          }
          await saveJson(path.join(output, 'ingestion.json'), {
            seconds: (performance.now() - started) / 1000,
            ...job,
          })
          break
        }
        if (job.state !== 'running') throw new Error(`Sync failed: ${JSON.stringify(job)}`)
        await new Promise((resolve) => setTimeout(resolve, 1000))
      }
    }
    // Warm up each new server session, including resumed sessions; exclude this from metrics.
    parseToolResult(await call('query_documents', { query: 'benchmark initialization', limit }))
    await fs.mkdir(path.join(output, 'queries'), { recursive: true })
    const known = new Set(dataset.corpus.map((doc) => doc._id))
    for (const [i, query] of dataset.queries.entries()) {
      const file = path.join(output, 'queries', `${query._id}.json`)
      if (!(await readJson(file))) {
        const args = { query: query.text, limit }
        const started = performance.now()
        const result = await call('query_documents', args)
        const milliseconds = performance.now() - started
        rankHits(result, environment.BASE_DIR, known, limit)
        await saveJson(file, { queryId: query._id, arguments: args, milliseconds, result })
      }
      if ((i + 1) % 50 === 0) console.log(`Queries: ${i + 1}/${dataset.queries.length}`)
    }
  } finally {
    await client.close()
  }
}

function tokenCounts(tokenizer, texts) {
  const counts = []
  for (let i = 0; i < texts.length; i += 128) {
    const encoded = tokenizer(texts.slice(i, i + 128), {
      padding: false,
      truncation: false,
      return_tensor: false,
    })
    counts.push(...encoded.input_ids.map((ids) => ids.length))
  }
  return counts
}

export async function summarize(output, config, dataset) {
  env.cacheDir = config.cacheDir
  const tokenizer = await AutoTokenizer.from_pretrained(config.modelName)
  const db = await connect(config.dbPath)
  const documentsDir = path.resolve(config.baseDirs[0])
  const table = await db.openTable('chunks')
  const rows = await table.query().select(['filePath', 'chunkIndex', 'text', 'vector']).toArray()
  const known = new Set(dataset.corpus.map((doc) => doc._id))
  const indexed = new Map()
  const dimension = rows[0]?.vector.length
  for (const row of rows) {
    if (
      path.dirname(row.filePath) !== documentsDir ||
      !known.has(path.basename(row.filePath, '.md'))
    ) {
      throw new Error('Index contains an unknown document')
    }
    const indices = indexed.get(row.filePath) ?? []
    indices.push(row.chunkIndex)
    indexed.set(row.filePath, indices)
    if (row.vector.length !== dimension || !Array.from(row.vector).every(Number.isFinite)) {
      throw new Error('Index contains invalid vectors')
    }
  }
  if (indexed.size !== known.size) throw new Error('Index does not cover the corpus')
  for (const indices of indexed.values()) {
    if (indices.sort((a, b) => a - b).some((index, i) => index !== i)) {
      throw new Error('Noncontinuous chunk indices')
    }
  }
  const details = []
  for (const query of dataset.queries) {
    const saved = await readJson(path.join(output, 'queries', `${query._id}.json`))
    if (saved.queryId !== query._id || saved.arguments.query !== query.text)
      throw new Error('Query mismatch')
    const hits = rankHits(saved.result, documentsDir, known, limit)
    const texts = saved.result.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
    details.push({
      queryId: query._id,
      query: query.text,
      milliseconds: saved.milliseconds,
      ...evaluate(hits, dataset.qrels[query._id], limit),
      chunkTextTokens: tokenCounts(
        tokenizer,
        hits.map((hit) => hit.text)
      ).reduce((a, b) => a + b, 0),
      mcpTextTokens: tokenCounts(tokenizer, texts).reduce((a, b) => a + b, 0),
      hits: hits.map(({ text, ...hit }) => ({ ...hit, characters: text.length })),
    })
  }
  const ingestion = await readJson(path.join(output, 'ingestion.json'))
  const summary = {
    documents: indexed.size,
    queries: details.length,
    chunks: rows.length,
    vectorDimension: dimension,
    ingestionSeconds: ingestion.seconds,
    chunkCharacters: distribution(rows.map((row) => row.text.length)),
    chunkTokens: distribution(
      tokenCounts(
        tokenizer,
        rows.map((row) => row.text)
      )
    ),
    queryMilliseconds: distribution(details.map((row) => row.milliseconds)),
  }
  for (const key of [
    'ndcgAt10',
    'chunkSlotNdcgAt10',
    'recallAt10',
    'chunkSlotMrrAt10',
    'relevantAt1',
    'uniqueDocuments',
    'chunkTextTokens',
    'mcpTextTokens',
  ]) {
    summary[key] = mean(details.map((row) => row[key]))
  }
  await saveJson(path.join(output, 'per-query.json'), details)
  await saveJson(path.join(output, 'summary.json'), summary)
  return summary
}
