import crypto from 'crypto';
import { stripStops } from './stopwords';

/**
 * Create a sparse vector from text with token frequencies
 * @param text Text to convert to sparse vector
 * @returns Sparse vector in Pinecone format (indices and values)
 */
export function toSparseVector(text: string) {
  const freq: Record<number, number> = {};
  
  // Split on whitespace after removing stop words
  for (const tok of stripStops(text).split(' ')) {
    if (!tok) continue; // Skip empty tokens
    
    // Hash the token to get a numeric ID
    const id = hash(tok);
    
    // Increment the frequency count
    freq[id] = (freq[id] ?? 0) + 1;
  }
  
  return {
    indices: Object.keys(freq).map(Number),
    values: Object.values(freq),
  };
}

/**
 * Create a 32-bit hash of a token
 * @param token Token to hash
 * @returns 32-bit numeric hash
 */
function hash(token: string): number {
  return crypto.createHash('md5').update(token).digest().readUInt32BE(0);
} 