import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import JSZip from 'jszip'

export const DATASET = {
  name: 'scifact',
  url: 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
  sha256: '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165',
}

export function sha256(data) {
  return createHash('sha256').update(data).digest('hex')
}

export async function saveJson(file, value) {
  await fs.writeFile(`${file}.tmp`, `${JSON.stringify(value, null, 2)}\n`)
  await fs.rename(`${file}.tmp`, file)
}

export async function readJson(file) {
  try {
    return JSON.parse(await fs.readFile(file, 'utf8'))
  } catch (error) {
    if (error.code === 'ENOENT') return undefined
    throw error
  }
}

function jsonLines(text) {
  return text
    .trim()
    .split(/\r?\n/)
    .map((line) => JSON.parse(line))
}

export async function readDataset(archive) {
  const zip = await JSZip.loadAsync(archive)
  const read = async (name) => {
    const entry = zip.file(`scifact/${name}`)
    if (!entry) throw new Error(`Missing dataset file: ${name}`)
    return entry.async('string')
  }
  const corpus = jsonLines(await read('corpus.jsonl'))
  const allQueries = jsonLines(await read('queries.jsonl'))
  const qrels = {}
  for (const line of (await read('qrels/test.tsv')).trim().split(/\r?\n/).slice(1)) {
    const [queryId, documentId, score] = line.split('\t')
    const relevance = Number(score)
    if (!queryId || !documentId || !Number.isFinite(relevance) || relevance < 0) {
      throw new Error(`Invalid qrel: ${line}`)
    }
    qrels[queryId] ??= {}
    qrels[queryId][documentId] = relevance
  }
  const documents = new Set(corpus.map((doc) => doc._id))
  for (const doc of corpus) {
    if (!/^[\w-]+$/.test(doc._id) || typeof doc.text !== 'string') {
      throw new Error('Invalid corpus document')
    }
  }
  const queries = allQueries.filter((query) => Object.hasOwn(qrels, query._id))
  if (
    documents.size !== corpus.length ||
    new Set(queries.map((q) => q._id)).size !== Object.keys(qrels).length
  ) {
    throw new Error('Duplicate documents or missing test queries')
  }
  for (const labels of Object.values(qrels)) {
    if (Object.keys(labels).some((id) => !documents.has(id))) {
      throw new Error('Qrels reference unknown documents')
    }
  }
  return { corpus, queries, qrels }
}

export async function prepareDataset(cacheDir, documentsDir) {
  await fs.mkdir(cacheDir, { recursive: true })
  const archivePath = path.join(cacheDir, 'scifact.zip')
  let archive
  try {
    archive = await fs.readFile(archivePath)
  } catch (error) {
    if (error.code !== 'ENOENT') throw error
    console.log(`Downloading ${DATASET.url}`)
    const response = await fetch(DATASET.url, { signal: AbortSignal.timeout(120_000) })
    if (!response.ok) throw new Error(`Dataset download failed: HTTP ${response.status}`)
    archive = Buffer.from(await response.arrayBuffer())
    if (sha256(archive) !== DATASET.sha256) throw new Error('Dataset checksum mismatch')
    await fs.writeFile(`${archivePath}.tmp`, archive)
    await fs.rename(`${archivePath}.tmp`, archivePath)
  }
  if (sha256(archive) !== DATASET.sha256)
    throw new Error(`Dataset checksum mismatch: ${archivePath}`)
  const dataset = await readDataset(archive)
  await fs.mkdir(documentsDir, { recursive: true })
  for (const doc of dataset.corpus) {
    const file = path.join(documentsDir, `${doc._id}.md`)
    const text = `# ${doc.title || 'none'}\n\n${doc.text}\n`
    let existing
    try {
      existing = await fs.readFile(file, 'utf8')
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
    }
    if (existing !== text) await fs.writeFile(file, text)
  }
  return dataset
}
