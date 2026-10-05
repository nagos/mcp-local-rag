import type { EmbeddingPipeline, PrefixPolicy } from '../types.js'

export const MINILM_MODEL = 'Xenova/all-MiniLM-L6-v2'

export const DEFAULT_PREFIX_POLICY: PrefixPolicy = {
  query: '',
  similarity: '',
  document: (title: string): string => `Title: ${title}\n\n`,
  fallback: '',
  titlePrefixEnv: true,
}

export async function runMeanInference(
  pipeline: EmbeddingPipeline,
  texts: string[]
): Promise<{ data?: unknown; dims?: unknown } | null | undefined> {
  return pipeline(texts, { pooling: 'mean', normalize: true })
}
