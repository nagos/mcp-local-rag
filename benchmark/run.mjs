import { execFileSync } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { DATASET, prepareDataset, readJson, saveJson, sha256 } from './dataset.mjs'

import { runMcp, summarize } from './mcp.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const { values } = parseArgs({ options: { output: { type: 'string' }, help: { type: 'boolean' } } })
const limit = 10

async function treeHash(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true })
  const hashes = []
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const file = path.join(directory, entry.name)
    hashes.push([
      entry.name,
      entry.isDirectory() ? await treeHash(file) : sha256(await fs.readFile(file)),
    ])
  }
  return sha256(JSON.stringify(hashes))
}

function serverEnvironment(output) {
  const environment = { ...process.env }
  for (const key of Object.keys(environment)) {
    if (key.startsWith('VLM_')) delete environment[key]
  }
  for (const key of [
    'BASE_DIRS',
    'RAG_GROUPING',
    'RAG_MAX_DISTANCE',
    'RAG_MAX_FILES',
    'RAG_RERANK_CMD',
    'RAG_RERANK_TIMEOUT_MS',
  ]) {
    delete environment[key]
  }
  return {
    ...environment,
    BASE_DIR: path.join(output, 'documents'),
    DB_PATH: path.join(output, 'index'),
    RAG_HYBRID_WEIGHT: '0.6',
    EMBED_TITLE_PREFIX: 'false',
    EMBED_HEADING_PREFIX: 'false',
    CHUNK_MIN_LENGTH: '50',
    MAX_FILE_SIZE: '104857600',
    STORE_IMAGES: 'false',
  }
}

async function initialize(output, environment) {
  const file = path.join(output, 'manifest.json')
  const previous = await readJson(file)
  if (!previous && (await fs.readdir(output)).some((name) => name !== '.lock')) {
    throw new Error(
      'Output directory is not empty and has no benchmark manifest; choose a new --output.'
    )
  }
  await fs.mkdir(environment.BASE_DIR, { recursive: true })
  const { resolveServerConfig } = await import('../dist/server-main.js')
  const config = await resolveServerConfig(environment, root)
  if (config.configError || config.configWarnings?.length) {
    throw new Error(`Invalid server configuration: ${JSON.stringify(config)}`)
  }
  config.cacheDir = path.resolve(root, config.cacheDir)
  const identity = {
    dataset: DATASET,
    limit,
    config,
    code: await treeHash(path.join(root, 'dist')),
    dependencies: sha256(await fs.readFile(path.join(root, 'pnpm-lock.yaml'))),
    harness: await Promise.all(
      ['run.mjs', 'dataset.mjs', 'metrics.mjs', 'mcp.mjs', 'report.mjs'].map(async (file) => [
        file,
        sha256(await fs.readFile(path.join(root, 'benchmark', file))),
      ])
    ),
  }
  if (previous) {
    if (JSON.stringify(previous.identity) !== JSON.stringify(identity)) {
      throw new Error(
        'Code or settings changed. Choose a new --output directory for a fresh index.'
      )
    }
    return previous
  }
  let commit = null
  try {
    commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim()
  } catch {
    // Source archives can be benchmarked without Git.
  }
  const manifest = {
    identity,
    commit,
    started: new Date().toISOString(),
    node: process.version,
    platform: `${os.platform()} ${os.release()} ${os.arch()}`,
    cpu: os.cpus()[0]?.model,
    logicalCpus: os.cpus().length,
    memoryBytes: os.totalmem(),
  }
  await saveJson(file, manifest)
  return manifest
}

async function main() {
  if (values.help) {
    console.log(
      'Usage: pnpm benchmark [--output DIRECTORY]\nRuns SciFact through the current MCP server. MODEL_NAME, CACHE_DIR, RAG_DEVICE and RAG_DTYPE are supported.\nModel and device settings are passed through to mcp-local-rag; hybrid weight 0.6; 10 returned chunks. Each new output gets a fresh index.'
    )
    return
  }
  const requestedOutput = path.resolve(
    values.output ||
      path.join(root, 'benchmark/runs', new Date().toISOString().replaceAll(':', '-'))
  )
  await fs.mkdir(requestedOutput, { recursive: true })
  const output = await fs.realpath(requestedOutput)
  const lockFile = path.join(output, '.lock')
  const lock = await fs.open(lockFile, 'wx')
  try {
    await lock.writeFile(`${process.pid}\n`)
    console.log('Building current code…')
    execFileSync('pnpm', ['run', 'build'], { cwd: root, stdio: 'inherit' })
    const environment = serverEnvironment(output)
    const manifest = await initialize(output, environment)
    const dataset = await prepareDataset(path.join(root, 'benchmark/data'), environment.BASE_DIR)
    if (!(await readJson(path.join(output, 'summary.json')))) {
      await runMcp(output, environment, dataset, root)
    }
    const summary = await summarize(output, manifest.identity.config, dataset)
    const { writeReport } = await import('./report.mjs')
    await writeReport(output, manifest, summary)
    console.log(JSON.stringify(summary, null, 2))
    console.log(`Report: ${path.join(output, 'REPORT.md')}`)
  } finally {
    await lock.close()
    await fs.unlink(lockFile)
  }
}

main().catch((error) => {
  console.error(error.message)
  process.exitCode = 1
})
