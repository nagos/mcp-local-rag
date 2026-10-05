import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

const MODEL = 'onnx-community/embeddinggemma-300m-ONNX'
let SemanticChunker: typeof import('../../chunker/index.js').SemanticChunker
let buildChunksAndEmbeddings: typeof import('../../ingest/compute.js').buildChunksAndEmbeddings
let Embedder: typeof import('../../embedder/index.js').Embedder
const pipeline = vi.fn()
const tokenizer = vi.fn((texts: string[], options: { max_length?: number }) => ({
  input_ids: texts.map((text) =>
    Array.from({ length: Math.min(text.length, options.max_length ?? Infinity) }, () => 1)
  ),
}))
const underlying = vi.fn(async (inputs: { input_ids: number[][] }): Promise<unknown> => {
  const data = new Float32Array(inputs.input_ids.length * 768)
  inputs.input_ids.forEach((ids, row) => {
    data[row * 768 + (ids.length % 768)] = 1
  })
  return { sentence_embedding: { dims: [inputs.input_ids.length, 768], data } }
})
const pooled = vi.fn(async (texts: string[]) => ({
  dims: [texts.length, 2],
  data: new Float32Array(texts.length * 2),
}))

function make(modelPath = MODEL, titlePrefix = false) {
  return new Embedder({ modelPath, titlePrefix, batchSize: 3, cacheDir: '/tmp/gemma-test' })
}

describe('EmbeddingGemma retrieval', () => {
  beforeAll(async () => {
    vi.resetModules()
    vi.doMock('@huggingface/transformers', () => ({ pipeline, env: {} }))
    ;({ Embedder } = await import('../../embedder/index.js'))
    ;({ SemanticChunker } = await import('../../chunker/index.js'))
    ;({ buildChunksAndEmbeddings } = await import('../../ingest/compute.js'))
  })
  beforeEach(() => {
    vi.clearAllMocks()
    pipeline.mockImplementation(async () =>
      Object.assign(pooled, {
        tokenizer: Object.assign(tokenizer, { model_max_length: 100 }),
        model: underlying,
      })
    )
  })
  afterAll(() => {
    vi.doUnmock('@huggingface/transformers')
    vi.resetModules()
  })

  it('selects native vectors, batches and restores deferred row order', async () => {
    const texts = ['abcde', 'abcdef', 'x'.repeat(90), 'hello']
    const vectors = await make().embedBatch(texts)
    expect(vectors.map((vector) => vector.indexOf(1))).toEqual(texts.map((text) => text.length))
    expect(underlying).toHaveBeenCalledTimes(3)
    expect(pooled).not.toHaveBeenCalled()
    expect(pipeline).toHaveBeenCalledOnce()
  })

  it('prepares single inputs once and leaves prepared document inputs unchanged', async () => {
    const embedder = make()
    for (const [text, type, prepared] of [
      ['cats', 'query', 'task: search result | query: cats'],
      ['cats', 'similarity', 'task: sentence similarity | query: cats'],
      ['title: none | text: cats', undefined, 'title: none | text: cats'],
    ] as const) {
      await embedder.embed(text, type)
      expect(tokenizer).toHaveBeenLastCalledWith(
        [prepared],
        expect.objectContaining({ padding: true, truncation: true })
      )
    }
  })

  it('automatically uses titles, none and the oversized-title fallback', async () => {
    const embedder = make()
    expect(await embedder.getDocumentPrefix('Cats')).toBe('title: Cats | text: ')
    expect(await embedder.getDocumentPrefix()).toBe('title: none | text: ')
    expect(await embedder.getDocumentPrefix('x'.repeat(100))).toBe('title: none | text: ')
    expect(await make(MODEL, true).getDocumentPrefix('Cats')).toBe('title: Cats | text: ')
  })

  it('enables the mode only for the exact model name and preserves legacy title behavior', async () => {
    const other = make(`${MODEL}-other`)
    await other.embed('cats', 'query')
    expect(pooled).toHaveBeenCalledWith(['cats'], { pooling: 'mean', normalize: true })
    expect(await other.getDocumentPrefix('Cats')).toBe('')
    expect(await make('other', true).getDocumentPrefix('Cats')).toBe('Title: Cats\n\n')
    expect(await make('other', true).getDocumentPrefix('x'.repeat(100))).toBe('')
  })

  it.each([
    {},
    { sentence_embedding: { dims: [1, 768], data: new Float32Array(768) } },
    { sentence_embedding: { dims: [1, 768], data: new Float32Array(768).fill(NaN) } },
    { sentence_embedding: { dims: [1, 384], data: new Float32Array(384) } },
    { sentence_embedding: { dims: [1, 1, 768], data: new Float32Array(768) } },
    { sentence_embedding: { dims: [2, 768], data: new Float32Array(1536) } },
    { sentence_embedding: { dims: [1, 768], data: new Float64Array(768) } },
  ])('rejects missing or invalid native output without pooling: %j', async (output) => {
    underlying.mockResolvedValueOnce(output)
    await expect(make().embed('cats')).rejects.toHaveProperty('name', 'EmbeddingError')
    expect(pooled).not.toHaveBeenCalled()
  })

  it('contains full document prefixes at the cap and uses similarity inputs and keeps stored text plain', async () => {
    const embedder = make()
    const prefix = 'title: Cats | text: '
    const text =
      'Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau.'
    const embedBatch = vi.spyOn(embedder, 'embedBatch')
    const result = await buildChunksAndEmbeddings(
      text,
      new SemanticChunker({ minChunkLength: 1 }),
      embedder,
      { title: 'Cats' }
    )
    expect(result.chunks.length).toBeGreaterThan(1)
    const lengths = await embedder.countTokens(result.chunks.map((chunk) => prefix + chunk.text))
    expect(lengths.every((length) => length <= 100)).toBe(true)
    expect(embedBatch.mock.calls[0]?.[1]).toBe('similarity')
    expect(result.chunks.every((chunk) => !chunk.text.includes('title:'))).toBe(true)
    expect(
      embedBatch.mock.calls[0]?.[0].every(
        (input) => !input.includes('task:') && !input.includes('title:')
      )
    ).toBe(true)
    expect(
      (await embedder.countTokens(embedBatch.mock.calls[0]?.[0] ?? [], 'similarity')).every(
        (length) => length <= 100
      )
    ).toBe(true)
    expect(embedBatch.mock.calls.at(-1)?.[0]).toEqual(
      result.chunks.map((chunk) => prefix + chunk.text)
    )
    expect(await embedder.countTokens([prefix + 'x'.repeat(100 - prefix.length)])).toEqual([100])
  })

  it('prepares typed inputs once and keeps empty batches lazy', async () => {
    const embedder = make()
    await expect(embedder.embedBatch([], 'similarity')).resolves.toEqual([])
    expect(pipeline).not.toHaveBeenCalled()
    await embedder.embedBatch(['cats', 'dogs'], 'similarity')
    expect(tokenizer).toHaveBeenCalledWith(
      ['task: sentence similarity | query: cats', 'task: sentence similarity | query: dogs'],
      expect.objectContaining({ padding: true, truncation: true })
    )
    expect(await embedder.countTokens(['cats'], 'similarity')).toEqual([39])
    await make('other').embedBatch(['cats'], 'similarity')
    expect(pooled).toHaveBeenCalledWith(['cats'], { pooling: 'mean', normalize: true })
  })

  it('clamps underlying tokenization while measurement remains unclamped', async () => {
    const embedder = make()
    await embedder.embed('x'.repeat(110))
    expect(await embedder.countTokens(['x'.repeat(110)])).toEqual([110])
    expect(underlying.mock.calls[0]?.[0].input_ids[0]).toHaveLength(100)
  })
})
