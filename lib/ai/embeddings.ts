import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

/**
 * Generate embeddings for a given text and pad to match Pinecone dimensions
 * @param text - The text to generate embeddings for
 * @returns An array of embeddings padded to 1024 dimensions
 */
export async function getEmbedding(text: string): Promise<number[]> {
  try {
    const embeddingModel = genAI.getGenerativeModel({ model: 'llama-text-embed-v2' });
    const result = await embeddingModel.embedContent(text);
    const embedding = result.embedding?.values || [];
    
    // Gemini returns 768-dim vectors, but Pinecone expects 1024-dim
    // Pad with zeros to reach 1024 dimensions
    if (embedding.length < 1024) {
      return [...embedding, ...Array(1024 - embedding.length).fill(0)];
    }
    
    return embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    throw error;
  }
} 