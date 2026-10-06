import assert from 'node:assert/strict'
import path from 'node:path'
import test from 'node:test'
import JSZip from 'jszip'
import { readDataset } from './dataset.mjs'
import { evaluate, parseToolResult, rankHits } from './metrics.mjs'

const hits = (...ids) => ids.map((documentId, index) => ({ documentId, rank: index + 1 }))

test('document metrics collapse duplicates, chunk-slot metrics preserve their positions', () => {
  const result = evaluate(hits('irrelevant', 'irrelevant', 'relevant'), { relevant: 1 })
  assert.equal(result.ndcgAt10, 1 / Math.log2(3))
  assert.equal(result.chunkSlotNdcgAt10, 0.5)
  assert.equal(result.chunkSlotMrrAt10, 1 / 3)
  assert.equal(result.recallAt10, 1)
  assert.equal(result.relevantAt1, 0)
})

test('graded relevance uses linear gain and recall includes unretrieved relevant documents', () => {
  const result = evaluate(hits('b', 'a', 'a'), { a: 2, b: 1, c: 1 })
  const ideal = 2 + 1 / Math.log2(3) + 0.5
  assert.equal(result.ndcgAt10, (1 + 2 / Math.log2(3)) / ideal)
  assert.equal(result.recallAt10, 2 / 3)
  assert.equal(result.relevantAt1, 1)
  assert.equal(result.uniqueDocuments, 2)
})

test('hits outside cutoff do not affect recall or reciprocal rank', () => {
  const result = evaluate(hits(...Array(10).fill('wrong'), 'relevant'), { relevant: 1 })
  assert.equal(result.ndcgAt10, 0)
  assert.equal(result.recallAt10, 0)
  assert.equal(result.chunkSlotMrrAt10, 0)
})

test('MCP errors and documents outside the isolated corpus fail evaluation', () => {
  assert.throws(() => parseToolResult({ isError: true, content: [] }), /MCP tool failed/)
  const directory = path.resolve('/tmp/benchmark-documents')
  const result = {
    content: [
      {
        type: 'text',
        text: JSON.stringify([
          { filePath: path.join(directory, '../elsewhere/a.md'), text: 'text' },
        ]),
      },
    ],
  }
  assert.throws(() => rankHits(result, directory, new Set(['a']), 10), /unknown document/)
})

test('BEIR archive selects test queries and preserves full document text', async () => {
  const zip = new JSZip()
  zip.file(
    'scifact/corpus.jsonl',
    JSON.stringify({ _id: 'doc', title: 'Title', text: 'Full text.\nSecond paragraph.' })
  )
  zip.file(
    'scifact/queries.jsonl',
    [
      { _id: 'train', text: 'Not evaluated' },
      { _id: 'test', text: 'Test query' },
    ]
      .map((query) => JSON.stringify(query))
      .join('\n')
  )
  zip.file('scifact/qrels/test.tsv', 'query-id\tcorpus-id\tscore\ntest\tdoc\t1\n')
  const result = await readDataset(await zip.generateAsync({ type: 'nodebuffer' }))
  assert.equal(result.corpus[0].text, 'Full text.\nSecond paragraph.')
  assert.deepEqual(
    result.queries.map((query) => query._id),
    ['test']
  )
  assert.deepEqual(result.qrels, { test: { doc: 1 } })
})
