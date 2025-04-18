import { getEmbedding } from './embeddings';

/**
 * Compute cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return normA && normB ? dot / (normA * normB) : 0;
}

/**
 * Estimate token count by word count × 1.3
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.split(/\s+/).length * 1.3);
}

/**
 * Split a paragraph into sentences
 */
function splitSentences(paragraph: string): string[] {
  const matches = paragraph.match(/[^.!?]+[.!?]+\s*|[^.!?]+$/g);
  return matches?.map(s => s.trim()) || [];
}

/**
 * Chunk text semantically by detecting topic shifts via embeddings
 * @param text Full document text
 * @param options
 *   - minTokens: minimum tokens per chunk before checking similarity (default: 500)
 *   - maxTokens: maximum tokens per chunk (default: 2000)
 *   - windowSize: number of sentences to consider in similarity window (default: 3)
 *   - threshold: cosine similarity cutoff for splitting (default: 0.75)
 * @returns Array of text chunks
 */
export async function chunkTextSemantic(
  text: string,
  {
    minTokens = 500,
    maxTokens = 2000,
    windowSize = 3,
    threshold = 0.75,
  }: {
    minTokens?: number;
    maxTokens?: number;
    windowSize?: number;
    threshold?: number;
  } = {}
): Promise<string[]> {
  // Split into paragraphs, then flatten into sentences
  const paragraphs = text.split(/\n{2,}/).map(p => p.trim()).filter(Boolean);
  const sentences = paragraphs.flatMap(splitSentences);

  const chunks: string[] = [];
  let currentSentences: string[] = [];
  let currentTokens = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    const sentTokens = estimateTokens(sentence);

    // If adding sentence exceeds maxTokens, force a split
    if (currentTokens + sentTokens > maxTokens) {
      chunks.push(currentSentences.join(' '));
      currentSentences = [];
      currentTokens = 0;
    }

    currentSentences.push(sentence);
    currentTokens += sentTokens;

    // Once we've reached the minimum size, check semantic shift
    const atLeastMin = currentTokens >= minTokens;
    const hasNextWindow = i + windowSize < sentences.length;

    if (atLeastMin && hasNextWindow) {
      // Build current window text and next window text
      const currentWindow = currentSentences.slice(-windowSize).join(' ');
      const nextWindow = sentences.slice(i + 1, i + 1 + windowSize).join(' ');

      try {
        const [embCurr, embNext] = await Promise.all([
          getEmbedding(currentWindow),
          getEmbedding(nextWindow),
        ]);
        const sim = cosineSimilarity(embCurr, embNext);

        if (sim < threshold) {
          // Topic shift detected → finalize chunk
          chunks.push(currentSentences.join(' '));
          currentSentences = [];
          currentTokens = 0;
        }
      } catch (err) {
        console.warn('Embedding error, skipping semantic split:', err);
        // On error, do not split; let size constraints handle it
      }
    }
  }

  // Push any remaining sentences
  if (currentSentences.length) {
    chunks.push(currentSentences.join(' '));
  }

  return chunks;
}
