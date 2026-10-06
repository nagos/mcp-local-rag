import path from 'node:path'

export function parseToolResult(result) {
  if (result.isError) throw new Error(`MCP tool failed: ${JSON.stringify(result)}`)
  for (const part of result.content ?? []) {
    if (part.type !== 'text') continue
    try {
      return JSON.parse(part.text)
    } catch {
      // Tools may include additional text blocks alongside their JSON response.
    }
  }
  throw new Error('MCP response has no JSON text block')
}

export function rankHits(result, documentsDir, knownDocuments, limit) {
  const hits = parseToolResult(result)
  if (!Array.isArray(hits) || hits.length === 0 || hits.length > limit) {
    throw new Error('Expected a nonempty chunk ranking within the requested cutoff')
  }
  return hits.map((hit, index) => {
    if (typeof hit.filePath !== 'string' || typeof hit.text !== 'string') {
      throw new Error('Invalid search hit')
    }
    const relative = path.relative(documentsDir, hit.filePath)
    const id = path.basename(relative, '.md')
    if (relative !== `${id}.md` || !knownDocuments.has(id)) {
      throw new Error(`Search returned an unknown document: ${hit.filePath}`)
    }
    return { documentId: id, rank: index + 1, chunkIndex: hit.chunkIndex, text: hit.text }
  })
}

function ndcg(ranking, labels, limit) {
  const gain = ranking
    .slice(0, limit)
    .reduce((sum, id, i) => sum + (labels[id] ?? 0) / Math.log2(i + 2), 0)
  const ideal = Object.values(labels)
    .sort((a, b) => b - a)
    .slice(0, limit)
    .reduce((sum, relevance, i) => sum + relevance / Math.log2(i + 2), 0)
  return ideal ? gain / ideal : 0
}

export function evaluate(hits, labels, limit = 10) {
  const ranking = [...new Set(hits.slice(0, limit).map((hit) => hit.documentId))]
  const relevant = ranking.filter((id) => (labels[id] ?? 0) > 0)
  const totalRelevant = Object.values(labels).filter((value) => value > 0).length
  const firstRelevant = hits.slice(0, limit).find((hit) => (labels[hit.documentId] ?? 0) > 0)
  const seen = new Set()
  const chunkSlots = hits.map((hit) => {
    const id = seen.has(hit.documentId) ? null : hit.documentId
    seen.add(hit.documentId)
    return id
  })
  return {
    ndcgAt10: ndcg(ranking, labels, limit),
    chunkSlotNdcgAt10: ndcg(chunkSlots, labels, limit),
    recallAt10: totalRelevant ? relevant.length / totalRelevant : 0,
    chunkSlotMrrAt10: firstRelevant ? 1 / firstRelevant.rank : 0,
    relevantAt1: Number(firstRelevant?.rank === 1),
    uniqueDocuments: ranking.length,
  }
}

export function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

export function distribution(values) {
  const sorted = [...values].sort((a, b) => a - b)
  if (!sorted.length) throw new Error('Cannot summarize empty measurements')
  return {
    mean: mean(sorted),
    median: sorted[Math.floor(sorted.length / 2)],
    p95: sorted[Math.floor(sorted.length * 0.95)],
    max: sorted.at(-1),
  }
}
