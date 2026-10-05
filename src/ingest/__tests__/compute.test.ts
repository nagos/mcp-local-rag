import { describe, expect, it, vi } from 'vitest'
import { asDouble } from '../../__tests__/test-doubles.js'
import type { SemanticChunker } from '../../chunker/index.js'
import type { EmbedderInterface } from '../../chunker/semantic-chunker.js'
import { buildChunksAndEmbeddings, buildChunksFromParseResult } from '../compute.js'

describe('buildChunksFromParseResult', () => {
  it('attaches a DOCX image to the chunk owning its source position', async () => {
    const text = 'Before text. After text.'
    const chunks = [
      { text: 'Before text.', index: 0, sourceStart: 0, sourceEnd: 11 },
      { text: 'After text.', index: 1, sourceStart: 13, sourceEnd: text.length },
    ]
    const chunkText = vi.fn().mockResolvedValue(chunks)
    const embedBatch = vi.fn().mockResolvedValue([
      [1, 0],
      [0, 1],
    ])
    const png = Uint8Array.from(
      Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
        'base64'
      )
    )

    const result = await buildChunksFromParseResult(
      {
        content: text,
        title: 'Document',
        imageAnchors: [{ offset: 12, imageIndex: 0, mimeType: 'image/png', bytes: png }],
      },
      asDouble<SemanticChunker>({ chunkText }),
      { embedBatch } satisfies EmbedderInterface
    )

    expect(result.visualAttachments.get(0)).toEqual([
      expect.objectContaining({ imageIndex: 0, mimeType: expect.stringMatching(/^image\//) }),
    ])
    expect(result.visualAttachments.has(1)).toBe(false)
  })
})

describe('buildChunksAndEmbeddings title prefix', () => {
  const chunks = () => [{ text: 'Body.', index: 0, sourceStart: 0, sourceEnd: 5 }]

  async function run(titlePrefix: boolean, embedder: Partial<EmbedderInterface> = {}) {
    const chunkText = vi.fn().mockResolvedValue(chunks())
    const embedBatch = vi.fn().mockResolvedValue([[1, 0]])
    const result = await buildChunksAndEmbeddings(
      'Body.',
      asDouble<SemanticChunker>({ chunkText }),
      {
        embedBatch,
        getDocumentPrefix: async () => (titlePrefix ? 'Title: Support Rotation\n\n' : ''),
        ...embedder,
      },
      { title: 'Support Rotation' }
    )
    return { result, chunkText, embedBatch }
  }

  it('embeds behind the title but keeps stored text body-only when enabled', async () => {
    const { result, chunkText, embedBatch } = await run(true)

    expect(chunkText.mock.calls[0]?.[3]).toBe('Title: Support Rotation\n\n')
    expect(embedBatch).toHaveBeenCalledWith(['Title: Support Rotation\n\nBody.'])
    expect(result.chunks[0]?.text).toBe('Body.')
  })

  it('embeds body text unchanged when disabled', async () => {
    const { chunkText, embedBatch } = await run(false)

    expect(chunkText.mock.calls[0]?.[3]).toBe('')
    expect(embedBatch).toHaveBeenCalledWith(['Body.'])
  })

  it('embeds behind the parser embedding title instead of the display title', async () => {
    const chunkText = vi.fn().mockResolvedValue(chunks())
    const embedBatch = vi.fn().mockResolvedValue([[1, 0]])

    await buildChunksFromParseResult(
      { content: 'Body.', title: 'Body text from page 1', embeddingTitle: 'quarterly report' },
      asDouble<SemanticChunker>({ chunkText }),
      { embedBatch, getDocumentPrefix: async (title) => `Title: ${title}\n\n` }
    )

    expect(embedBatch).toHaveBeenCalledWith(['Title: quarterly report\n\nBody.'])
  })

  it('delegates the title fallback to getDocumentPrefix', async () => {
    const getDocumentPrefix = vi.fn().mockResolvedValue('')
    const { embedBatch } = await run(true, { getDocumentPrefix })
    expect(getDocumentPrefix).toHaveBeenCalledWith('Support Rotation')
    expect(embedBatch).toHaveBeenCalledWith(['Body.'])
  })
})

describe('buildChunksAndEmbeddings heading prefix', () => {
  async function run(headingPrefix: boolean, cap?: number) {
    const text = 'Restore the previous version.'
    const chunkText = vi
      .fn()
      .mockResolvedValue([{ text, index: 0, sourceStart: 0, sourceEnd: text.length }])
    const embedBatch = vi.fn().mockResolvedValue([[1, 0]])
    const result = await buildChunksAndEmbeddings(
      text,
      asDouble<SemanticChunker>({ chunkText }),
      {
        embedBatch,
        headingPrefix,
        ...(cap === undefined
          ? {}
          : {
              getTokenLimit: async () => cap,
              countTokens: async (texts: string[]) => texts.map((value) => value.length),
            }),
      },
      {
        sourceMap: {
          headings: [
            { offset: 0, level: 1, text: 'Deployment' },
            { offset: 0, level: 2, text: 'Rollback' },
          ],
        },
      }
    )
    return { result, embedBatch }
  }

  it('skips heading token counting when no chunk has heading context', async () => {
    const text = 'Body.'
    const countTokens = vi.fn(async (texts: string[]) => texts.map(() => 1))
    const embedBatch = vi.fn().mockResolvedValue([[1, 0]])
    await buildChunksAndEmbeddings(
      text,
      asDouble<SemanticChunker>({
        chunkText: async () => [{ text, index: 0, sourceStart: 0, sourceEnd: text.length }],
      }),
      { headingPrefix: true, getTokenLimit: async () => 512, countTokens, embedBatch },
      { sourceMap: { headings: [] } }
    )
    expect(embedBatch).toHaveBeenCalledWith([text])
    expect(countTokens).not.toHaveBeenCalled()
  })

  it('adds section context only to embedding input when enabled', async () => {
    const { result, embedBatch } = await run(true)
    expect(embedBatch).toHaveBeenCalledWith([
      'Section: Deployment > Rollback\n\nRestore the previous version.',
    ])
    expect(result.chunks[0]?.text).toBe('Restore the previous version.')
    expect(result.chunks[0]?.sourceContext?.headingPaths).toEqual([['Deployment', 'Rollback']])
  })

  it('keeps metadata without changing embeddings when disabled', async () => {
    const { result, embedBatch } = await run(false)
    expect(embedBatch).toHaveBeenCalledWith(['Restore the previous version.'])
    expect(result.chunks[0]?.sourceContext?.headingPaths).toEqual([['Deployment', 'Rollback']])
  })

  it.each([40, 60])('omits headings instead of truncating body at cap %i', async (cap) => {
    const { embedBatch } = await run(true, cap)
    expect(embedBatch).toHaveBeenCalledWith(['Restore the previous version.'])
  })
  it('keeps the title and full body when additional headings overflow the input', async () => {
    const text = 'A sufficiently long body that must survive unchanged.'
    const embedBatch = vi.fn().mockResolvedValue([[1, 0]])
    await buildChunksAndEmbeddings(
      text,
      asDouble<SemanticChunker>({
        chunkText: async () => [{ text, index: 0, sourceStart: 0, sourceEnd: text.length }],
      }),
      {
        embedBatch,
        getDocumentPrefix: async (title) => `Title: ${title}\n\n`,
        headingPrefix: true,
        getTokenLimit: async () => 80,
        countTokens: async (texts) => texts.map((value) => value.length),
      },
      { title: 'Guide', sourceMap: { headings: [{ offset: 0, level: 1, text: 'Deploy' }] } }
    )
    expect(embedBatch).toHaveBeenCalledWith([`Title: Guide\n\n${text}`])
  })

  it('combines title and each intersecting heading path when they fit', async () => {
    const text = 'Body. More.'
    const embedBatch = vi.fn().mockResolvedValue([[1, 0]])
    await buildChunksAndEmbeddings(
      text,
      asDouble<SemanticChunker>({
        chunkText: async () => [{ text, index: 0, sourceStart: 0, sourceEnd: text.length }],
      }),
      {
        embedBatch,
        getDocumentPrefix: async (title) => `Title: ${title}\n\n`,
        headingPrefix: true,
      },
      {
        title: 'Guide',
        sourceMap: {
          headings: [
            { offset: 0, level: 1, text: 'Deploy' },
            { offset: 6, level: 1, text: 'Verify' },
          ],
        },
      }
    )
    expect(embedBatch).toHaveBeenCalledWith([
      `Title: Guide\n\nSection: Deploy\nSection: Verify\n\n${text}`,
    ])
  })
})
