import { AppError } from '../utils/errors.js'

export class EmbeddingError extends AppError {
  constructor(message: string, options?: { cause?: Error }) {
    super(message, 'embedder', 'internal', options)
    this.name = 'EmbeddingError'
  }
}
