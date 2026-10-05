import { parseBooleanEnv } from '../cli/options.js'
import { BGE_SMALL_EN_MODEL, BGE_SMALL_EN_PREFIX_POLICY } from './models/bge.js'
import { EMBEDDING_GEMMA_MODEL, EMBEDDING_GEMMA_PREFIX_POLICY } from './models/gemma.js'
import { DEFAULT_PREFIX_POLICY, MINILM_MODEL } from './models/minilm.js'
import type { EmbeddingType, PrefixPolicy } from './types.js'

const policies = new Map<string, PrefixPolicy>([
  [MINILM_MODEL, DEFAULT_PREFIX_POLICY],
  [BGE_SMALL_EN_MODEL, BGE_SMALL_EN_PREFIX_POLICY],
  [EMBEDDING_GEMMA_MODEL, EMBEDDING_GEMMA_PREFIX_POLICY],
])

export function resolveTitlePrefixEnv(
  modelPath: string,
  raw: string | undefined
): { value: boolean; warning?: string } {
  const policy = policies.get(modelPath) ?? DEFAULT_PREFIX_POLICY
  return policy.titlePrefixEnv ? parseBooleanEnv('EMBED_TITLE_PREFIX', raw) : { value: false }
}

export function getInputPrefix(modelPath: string, type?: EmbeddingType): string {
  const policy = policies.get(modelPath)
  if (!type || !policy) {
    return ''
  }
  return type === 'similarity' ? (policy.similarity ?? policy.query) : policy.query
}

interface PrefixMeasurement {
  cap: number | null
  countTokens(texts: string[]): Promise<number[]>
}

/** Oversized headers fall back to the model's document format without a title. */
export async function resolveDocumentPrefix(
  modelPath: string,
  titlePrefix: boolean,
  title: string | null | undefined,
  measure: () => Promise<PrefixMeasurement>
): Promise<string> {
  const policy = policies.get(modelPath) ?? DEFAULT_PREFIX_POLICY
  if (policy.titlePrefixEnv && (!titlePrefix || !title)) {
    return ''
  }
  const { document: format, fallback } = policy
  if (!title) {
    return fallback
  }
  const prefix = format(title)
  if (prefix.length === 0) {
    return ''
  }
  const measurement = await measure()
  const { cap } = measurement
  if (cap === null) {
    return prefix
  }
  const [length] = await measurement.countTokens([prefix])
  if (length === undefined) {
    throw new Error('Missing prefix token measurement')
  }
  return length <= cap / 2 ? prefix : fallback
}
