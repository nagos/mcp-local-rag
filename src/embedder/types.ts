export type EmbeddingType = 'query' | 'similarity'

export interface PrefixPolicy {
  query: string
  similarity?: string
  document(title: string): string
  fallback: string
  titlePrefixEnv?: boolean
}

interface TokenizerOptions {
  padding?: boolean
  truncation?: boolean
  return_tensor?: boolean
  max_length?: number
}

export interface PipelineTokenizer {
  (input: string[], options: TokenizerOptions): { input_ids?: unknown } | null | undefined
  /** `unknown` because a model may omit it or report a sentinel; sentinel validation remains in Embedder. */
  model_max_length?: unknown
}

interface PipelineModelConfig {
  max_position_embeddings?: unknown
}

/**
 * The transformers.js pipeline as this module calls it.
 *
 * Every result field is typed as loosely as the runtime admits — `dims`
 * included, since the runtime guard establishes only that the value
 * is callable and carries a `tokenizer`. That keeps each call site's own shape
 * check load-bearing rather than dead under an optimistic declaration.
 */
export interface EmbeddingPipeline {
  (
    input: string[],
    options: unknown
  ): Promise<{ data?: unknown; dims?: unknown } | null | undefined>
  tokenizer: PipelineTokenizer
  /** Optional: a model reporting no position window takes the tokenizer-only branch. */
  model?: { config?: PipelineModelConfig } | UnderlyingEmbeddingModel
}

interface UnderlyingEmbeddingModel {
  (inputs: unknown): Promise<unknown>
  config?: PipelineModelConfig
}
