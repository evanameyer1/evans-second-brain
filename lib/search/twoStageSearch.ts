/**
 * Two-Stage Search Implementation
 *
 * Stage 1: Header Search (Coarse Filtering)
 * - Search against document headers to quickly narrow down candidates
 *
 * Stage 2: Body Chunk Search (Fine-Grained Ranking)
 * - Search against chunks from candidate documents for precise results
 */

import { Pinecone } from '@pinecone-database/pinecone';
import { stripStops } from '../ai/stopwords';
import { toSparseVector } from '../ai/sparse';
import { hybridSearch, formatReadwiseContext } from '../ai/readwise-search';
import { generateEmbeddings } from '../ai/embeddings';

// Initialize Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
});
const index = pc.index(process.env.PINECONE_INDEX || 'reader-embeddings');

/**
 * Process user query by removing stopwords
 * @param query User query
 * @returns Processed query
 */
function processQuery(query: string): string {
  // Remove stopwords and normalize
  return stripStops(query.toLowerCase().trim());
}

/**
 * Stage 1: Header Search - Find candidate documents based on headers
 * @param query User query
 * @param limit Maximum number of candidate documents to return
 * @returns Array of candidate document IDs with scores
 */
export async function headerSearch(
  query: string,
  limit = 10,
): Promise<Array<{ docId: string; score: number }>> {
  // Process query
  const processedQuery = processQuery(query);

  // Generate query embedding
  const queryEmbeddings = await generateEmbeddings(processedQuery, 1536);

  // Extract vector values from embedding
  const queryVector = Array.isArray(queryEmbeddings[0])
    ? (queryEmbeddings[0] as number[])
    : queryEmbeddings[0] &&
        typeof queryEmbeddings[0] === 'object' &&
        'values' in queryEmbeddings[0]
      ? (queryEmbeddings[0] as { values: number[] }).values
      : [];

  // Create sparse vector for hybrid search
  const sparseVector = toSparseVector(processedQuery);

  // Search headers with hybrid search
  const response = await index.query({
    vector: queryVector,
    sparseVector: sparseVector,
    topK: limit,
    filter: { header: true },
    includeMetadata: true,
  });

  // Extract document IDs and scores
  const candidates = (response.matches || []).map((match) => ({
    docId: (match.metadata?.doc_id as string) || '',
    score: match.score || 0,
  }));

  console.log(`Stage 1: Found ${candidates.length} candidate documents`);

  return candidates;
}

/**
 * Stage 2: Body Chunk Search - Find specific chunks within candidate documents
 * @param query User query
 * @param candidateDocIds Array of candidate document IDs from Stage 1
 * @param limit Maximum number of chunks to return
 * @returns Array of text chunks with metadata
 */
export async function bodyChunkSearch(
  query: string,
  candidateDocIds: string[],
  limit = 5,
): Promise<
  Array<{
    text: string;
    docId: string;
    title: string;
    url: string;
    score: number;
  }>
> {
  if (candidateDocIds.length === 0) {
    console.log('No candidate documents provided for Stage 2');
    return [];
  }

  // Process query
  const processedQuery = processQuery(query);

  // Generate query embedding
  const queryEmbeddings = await generateEmbeddings(processedQuery, 1536);

  // Extract vector values from embedding
  const queryVector = Array.isArray(queryEmbeddings[0])
    ? (queryEmbeddings[0] as number[])
    : queryEmbeddings[0] &&
        typeof queryEmbeddings[0] === 'object' &&
        'values' in queryEmbeddings[0]
      ? (queryEmbeddings[0] as { values: number[] }).values
      : [];

  // Create sparse vector for hybrid search
  const sparseVector = toSparseVector(processedQuery);

  // Search chunks with hybrid search, filtering by candidate document IDs
  const response = await index.query({
    vector: queryVector,
    sparseVector: sparseVector,
    topK: limit,
    filter: {
      header: false,
      doc_id: { $in: candidateDocIds },
    },
    includeMetadata: true,
  });

  // Extract chunks with metadata
  const chunks = (response.matches || []).map((match) => ({
    text: (match.metadata?.text as string) || '',
    docId: (match.metadata?.doc_id as string) || '',
    title: (match.metadata?.title as string) || '',
    url: (match.metadata?.url as string) || '',
    score: match.score || 0,
  }));

  console.log(`Stage 2: Found ${chunks.length} relevant chunks`);

  return chunks;
}

/**
 * Complete two-stage search process
 * @param query User query
 * @param headerLimit Maximum number of candidates from Stage 1
 * @param chunkLimit Maximum number of chunks to return from Stage 2
 * @returns Array of text chunks with metadata
 */
export async function twoStageSearch(
  query: string,
  headerLimit = 10,
  chunkLimit = 5,
): Promise<
  Array<{
    text: string;
    docId: string;
    title: string;
    url: string;
    score: number;
  }>
> {
  console.log(`Executing two-stage search for query: "${query}"`);

  // Stage 1: Find candidate documents based on header search
  const candidates = await headerSearch(query, headerLimit);

  // Extract document IDs
  const candidateDocIds = candidates.map((c) => c.docId);

  if (candidateDocIds.length === 0) {
    console.log('No candidate documents found in Stage 1');
    return [];
  }

  // Stage 2: Search within candidate documents' chunks
  const results = await bodyChunkSearch(query, candidateDocIds, chunkLimit);

  return results;
}

/**
 * Enhanced two-stage search using Gemini query optimization
 * @param query User query
 * @param limit Maximum number of chunks to return
 * @returns Array of text chunks with metadata
 */
export async function enhancedTwoStageSearch(
  query: string,
  limit = 5,
): Promise<
  Array<{
    text: string;
    docId: string;
    title: string;
    url: string;
    score: number;
  }>
> {
  console.log(`Executing enhanced two-stage search for query: "${query}"`);

  // Use the enhanced hybrid search which now includes Gemini query optimization
  try {
    console.log('Calling hybridSearch with Gemini enhancement...');
    const matches = await hybridSearch(query, limit, 0.7);

    console.log(`\n===== HYBRID SEARCH RESULTS SUMMARY =====`);
    console.log(`Received ${matches.length} matches from hybridSearch`);

    if (matches.length === 0) {
      console.log('No relevant content found for the query');
      return [];
    }

    // Log the matches we received
    matches.forEach((match, i) => {
      console.log(`\nMatch ${i + 1}:`);
      console.log(`Title: ${match.title}`);
      console.log(`Score: ${match.score.toFixed(4)}`);
      console.log(`Text preview: ${match.text.substring(0, 50)}...`);
    });
    console.log(`=========================================\n`);

    // Transform the results to match our expected format
    const results = matches.map((match) => ({
      text: match.text,
      docId: '', // This information is not available in the hybridSearch response
      title: match.title,
      url: '', // This information is not available in the hybridSearch response
      score: match.score,
    }));

    console.log(`Returning ${results.length} formatted results`);
    return results;
  } catch (error) {
    console.error('Error in enhanced two-stage search:', error);
    if (error instanceof Error) {
      console.error('Error details:', error.message);
      console.error('Error stack:', error.stack);
    }
    return [];
  }
}
