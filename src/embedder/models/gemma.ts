import { isObjectLike } from '../../utils/type-guards.js'
import { EmbeddingError } from '../errors.js'
import type { EmbeddingPipeline, PrefixPolicy } from '../types.js'

export const EMBEDDING_GEMMA_MODEL = 'onnx-community/embeddinggemma-300m-ONNX'

// Task prompts: https://ai.google.dev/gemma/docs/embeddinggemma/model_card#prompt-instructions
export const EMBEDDING_GEMMA_PREFIX_POLICY: PrefixPolicy = {
  query: 'task: search result | query: ',
  similarity: 'task: sentence similarity | query: ',
  document: (title: string): string => `title: ${title} | text: `,
  fallback: 'title: none | text: ',
}

function validateGemmaVector(vector: Float32Array): void {
  let squaredNorm = 0
  for (const value of vector) {
    if (!Number.isFinite(value)) {
      throw new EmbeddingError('Invalid EmbeddingGemma sentence_embedding: non-finite vector')
    }
    squaredNorm += value * value
  }
  if (Math.abs(Math.sqrt(squaredNorm) - 1) > 1e-3) {
    throw new EmbeddingError(
      'Invalid EmbeddingGemma sentence_embedding: expected normalized vector'
    )
  }
}

export async function runGemmaInference(
  pipeline: EmbeddingPipeline,
  texts: string[]
): Promise<{ data: Float32Array; dims: number[] }> {
  const model = pipeline.model
  if (typeof model !== 'function') {
    throw new EmbeddingError('EmbeddingGemma underlying model is not callable')
  }
  const tokenized = pipeline.tokenizer(texts, { padding: true, truncation: true })
  const result = await Promise.resolve(model(tokenized))
  const sentence = isObjectLike(result) ? result['sentence_embedding'] : undefined
  if (!isObjectLike(sentence)) {
    throw new EmbeddingError('Missing EmbeddingGemma sentence_embedding output')
  }
  const data = sentence['data']
  const dims = sentence['dims']
  if (
    !(data instanceof Float32Array) ||
    !Array.isArray(dims) ||
    dims.length !== 2 ||
    dims[0] !== texts.length ||
    dims[1] !== 768 ||
    data.length !== texts.length * 768
  ) {
    throw new EmbeddingError(
      'Unexpected EmbeddingGemma sentence_embedding shape (expected [batch, 768])'
    )
  }
  for (let row = 0; row < texts.length; row++) {
    validateGemmaVector(data.subarray(row * 768, (row + 1) * 768))
  }
  return { data, dims: [texts.length, 768] }
}
