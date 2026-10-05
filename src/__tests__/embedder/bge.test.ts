import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { MINILM_MODEL } from '../../embedder/models/minilm.js'
import {
  getInputPrefix,
  resolveDocumentPrefix,
  resolveTitlePrefixEnv,
} from '../../embedder/prefixes.js'

const MODEL = 'Xenova/bge-small-en-v1.5'
const PREFIX = 'Represent this sentence for searching relevant passages: '
let Embedder: typeof import('../../embedder/index.js').Embedder
const pipeline = vi.fn()
const tokenizer = vi.fn((texts: string[], options: { max_length?: number }) => ({
  input_ids: texts.map((text) =>
    Array.from({ length: Math.min(text.length, options.max_length ?? Infinity) }, () => 1)
  ),
}))
const pooled = vi.fn(async (texts: string[]) => {
  const data = new Float32Array(texts.length * 384)
  texts.forEach((text, index) => {
    data[index * 384 + (text.length % 384)] = 1
  })
  return { dims: [texts.length, 384], data }
})
function make(modelPath = MODEL, titlePrefix = false) {
  return new Embedder({ modelPath, titlePrefix, batchSize: 3, cacheDir: '/tmp/bge-test' })
}

describe('BGE small policy', () => {
  beforeAll(async () => {
    vi.resetModules()
    vi.doMock('@huggingface/transformers', () => ({ pipeline, env: {} }))
    ;({ Embedder } = await import('../../embedder/index.js'))
  })
  beforeEach(() => {
    vi.clearAllMocks()
    pipeline.mockResolvedValue(
      Object.assign(pooled, { tokenizer: Object.assign(tokenizer, { model_max_length: 100 }) })
    )
  })
  afterAll(() => {
    vi.doUnmock('@huggingface/transformers')
    vi.resetModules()
  })

  it('registers the exact model and ignores title flags without tokenizer access', async () => {
    expect(getInputPrefix(MODEL, 'query')).toBe(PREFIX)
    expect(getInputPrefix(MODEL, 'similarity')).toBe('')
    expect(getInputPrefix(`${MODEL}-other`, 'query')).toBe('')
    expect(resolveTitlePrefixEnv(MODEL, 'invalid')).toEqual({ value: false })
    const measure = vi.fn()
    expect(await resolveDocumentPrefix(MODEL, true, 'Title', measure)).toBe('')
    expect(measure).not.toHaveBeenCalled()
    expect(await make(MODEL, true).getDocumentPrefix('Title')).toBe('')
    expect(pipeline).not.toHaveBeenCalled()
  })

  it('uses CLS and adds the prefix only to query input', async () => {
    const embedder = make()
    await embedder.embed('cats', 'query')
    expect(pooled).toHaveBeenLastCalledWith([`${PREFIX}cats`], { pooling: 'cls', normalize: true })
    await embedder.embedBatch(['cats'], 'similarity')
    expect(pooled).toHaveBeenLastCalledWith(['cats'], { pooling: 'cls', normalize: true })
    await embedder.embedBatch([`${PREFIX}cats`])
    expect(pooled).toHaveBeenLastCalledWith([`${PREFIX}cats`], { pooling: 'cls', normalize: true })
    for (const name of [MINILM_MODEL, `${MODEL}-other`]) {
      await make(name).embed('cats', 'query')
      expect(pooled).toHaveBeenLastCalledWith(['cats'], { pooling: 'mean', normalize: true })
    }
  })

  it('keeps batching, deferred order, lazy empty input and empty text validation', async () => {
    const embedder = make()
    await expect(embedder.embedBatch([], 'query')).resolves.toEqual([])
    expect(pipeline).not.toHaveBeenCalled()
    await expect(embedder.embedBatch([''], 'query')).rejects.toHaveProperty(
      'name',
      'EmbeddingError'
    )
    const texts = ['short', 'tiny', 'x'.repeat(90), 'last']
    const vectors = await embedder.embedBatch(texts)
    expect(vectors.map((vector) => vector.indexOf(1))).toEqual(texts.map((text) => text.length))
    expect(pooled).toHaveBeenCalledTimes(3)
    expect(pipeline).toHaveBeenCalledOnce()
  })

  it('measures the full query including its instruction at the token boundary', async () => {
    const embedder = make()
    const atCap = 'x'.repeat(100 - PREFIX.length)
    expect(await embedder.countTokens([atCap], 'query')).toEqual([100])
    expect(await embedder.countTokens([`${atCap}x`], 'query')).toEqual([101])
    await embedder.embed(`${atCap}x`, 'query')
    expect(pooled).toHaveBeenLastCalledWith([`${PREFIX}${atCap}x`], {
      pooling: 'cls',
      normalize: true,
    })
  })
})
