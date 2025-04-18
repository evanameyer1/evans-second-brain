import crypto from 'node:crypto';
import { stripStops } from './stopwords';

/**
 * Create a sparse vector from text with token frequencies,
 * but only keep the top `maxTerms` most frequent tokens.
 *
 * @param text     Text to convert to sparse vector
 * @param maxTerms Maximum number of token-terms to include (default: 1536)
 * @returns        Sparse vector in Pinecone format (indices and values)
 */
export function toSparseVector(
  text: string,
  maxTerms = 1536,
): { indices: number[]; values: number[] } {
  const freq: Record<number, number> = {};

  // Build raw token frequencies
  for (const tok of stripStops(text).split(/\s+/)) {
    if (!tok) continue;
    const id = hash(tok);
    freq[id] = (freq[id] ?? 0) + 1;
  }

  // Sort entries by descending count, take top maxTerms
  const top = Object.entries(freq)
    .map(([id, count]) => ({ id: Number(id), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, maxTerms);

  // Split into parallel arrays
  const indices = top.map((e) => e.id);
  const values = top.map((e) => e.count);

  return { indices, values };
}

/**
 * Create a 32-bit hash of a token
 * @param token Token to hash
 * @returns 32-bit numeric hash
 */
function hash(token: string): number {
  return crypto.createHash('md5').update(token).digest().readUInt32BE(0);
}
