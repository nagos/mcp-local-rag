import { describe, expect, it } from 'vitest'
import { MINILM_MODEL } from '../../embedder/models/minilm.js'
import {
  getInputPrefix,
  resolveDocumentPrefix,
  resolveTitlePrefixEnv,
} from '../../embedder/prefixes.js'

const model = 'onnx-community/embeddinggemma-300m-ONNX'
function measurement(cap: number | null) {
  return async () => ({
    cap,
    countTokens: async (texts: string[]) => texts.map((text) => [...text].length + 2),
  })
}

describe('model prefix policies', () => {
  it('resolves only exact known names, including inherited object keys', () => {
    expect(getInputPrefix(model, 'query')).toBe('task: search result | query: ')
    expect(getInputPrefix(model, 'similarity')).toBe('task: sentence similarity | query: ')
    expect(getInputPrefix(`${model}-other`, 'query')).toBe('')
    expect(getInputPrefix('toString', 'query')).toBe('')
    expect(resolveTitlePrefixEnv(model, 'invalid')).toEqual({ value: false })
    expect(resolveTitlePrefixEnv('other', 'on')).toEqual({ value: true })
    expect(resolveTitlePrefixEnv('other', 'invalid').warning).toContain(
      'Invalid EMBED_TITLE_PREFIX'
    )
  })
  it.each([MINILM_MODEL, 'other'])(
    'preserves default prefixes and title settings for %s',
    async (name) => {
      expect(getInputPrefix(name, 'query')).toBe('')
      expect(getInputPrefix(name, 'similarity')).toBe('')
      expect(resolveTitlePrefixEnv(name, 'on')).toEqual({ value: true })
      expect(resolveTitlePrefixEnv(name, 'invalid').warning).toContain('Invalid EMBED_TITLE_PREFIX')
      expect(await resolveDocumentPrefix(name, false, 'Cats', measurement(100))).toBe('')
      expect(await resolveDocumentPrefix(name, true, 'Cats', measurement(100))).toBe(
        'Title: Cats\n\n'
      )
      expect(await resolveDocumentPrefix(name, true, undefined, measurement(100))).toBe('')
      expect(await resolveDocumentPrefix(name, true, 'x'.repeat(100), measurement(100))).toBe('')
    }
  )
  it('keeps titles up to half the window and drops oversized ones', async () => {
    expect(await resolveDocumentPrefix(model, false, 'x'.repeat(32), measurement(100))).toBe(
      `title: ${'x'.repeat(32)} | text: `
    )
    expect(await resolveDocumentPrefix(model, false, 'x'.repeat(33), measurement(100))).toBe(
      'title: none | text: '
    )
    expect(await resolveDocumentPrefix(model, false, undefined, measurement(100))).toBe(
      'title: none | text: '
    )
    expect(await resolveDocumentPrefix('other', true, 'abc', measurement(10))).toBe('')
    expect(await resolveDocumentPrefix('other', true, 'x'.repeat(600), measurement(null))).toBe(
      `Title: ${'x'.repeat(600)}\n\n`
    )
    expect(await resolveDocumentPrefix('other', false, 'abc', measurement(100))).toBe('')
    expect(getInputPrefix(model)).toBe('')
  })
})
