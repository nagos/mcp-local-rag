import { BGE_SMALL_EN_MODEL, runClsInference } from './models/bge.js'
import { EMBEDDING_GEMMA_MODEL, runGemmaInference } from './models/gemma.js'
import { MINILM_MODEL, runMeanInference } from './models/minilm.js'
import type { EmbeddingPipeline } from './types.js'

export type EmbeddingInference = (
  pipeline: EmbeddingPipeline,
  texts: string[]
) => Promise<{ data?: unknown; dims?: unknown } | null | undefined>

const inferenceModels = new Map<string, EmbeddingInference>([
  // Output: https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX#usage
  [EMBEDDING_GEMMA_MODEL, runGemmaInference],
  // Pooling: https://huggingface.co/BAAI/bge-small-en-v1.5#usage-for-embedding-model
  [BGE_SMALL_EN_MODEL, runClsInference],
  [MINILM_MODEL, runMeanInference],
])

export function resolveInference(modelPath: string): EmbeddingInference {
  return inferenceModels.get(modelPath) ?? runMeanInference
}
