import type { EmbeddingPipeline, PrefixPolicy } from '../types.js'

export const BGE_SMALL_EN_MODEL = 'Xenova/bge-small-en-v1.5'

// Query instructions: https://huggingface.co/BAAI/bge-small-en-v1.5#usage-for-embedding-model
export const BGE_SMALL_EN_PREFIX_POLICY: PrefixPolicy = {
  query: 'Represent this sentence for searching relevant passages: ',
  similarity: '',
  document: (): string => '',
  fallback: '',
}

export async function runClsInference(
  pipeline: EmbeddingPipeline,
  texts: string[]
): Promise<{ data?: unknown; dims?: unknown } | null | undefined> {
  return pipeline(texts, { pooling: 'cls', normalize: true })
}
