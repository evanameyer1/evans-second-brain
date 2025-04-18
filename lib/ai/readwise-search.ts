import 'dotenv/config';
import { Pinecone } from '@pinecone-database/pinecone';
import { stripStops } from './stopwords';
import { toSparseVector } from './sparse';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { generateEmbeddings } from './embeddings';

// Initialize Google Generative AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// Log environment variable status for debugging
if (!process.env.GEMINI_API_KEY) {
  console.error('GEMINI_API_KEY is missing in environment');
}

if (!process.env.PINECONE_API_KEY) {
  console.error('PINECONE_API_KEY is missing in environment');
}

if (!process.env.PINECONE_INDEX) {
  console.error('PINECONE_INDEX is missing in environment');
}

// Initialize Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
});

// Get index
const index = pc.index(process.env.PINECONE_INDEX || 'reader-embeddings');

// Helper function for timestamped logging
function logWithTimestamp(message: string, data?: any) {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
  if (data) {
    console.log(`[${timestamp}] Data:`, data);
  }
}

interface ReadwiseMatch {
  score: number;
  title: string;
  text: string;
  docId?: string;
  url?: string;
}

/**
 * Use Gemini to enhance a search query for better semantic matching
 * @param rawQuery User's original query
 * @returns Enhanced version of the query optimized for vector search
 */
async function generateEnhancedQuery(rawQuery: string): Promise<string> {
  try {
    logWithTimestamp('Starting query enhancement with Gemini', { rawQuery });

    // Create the prompt template
    const prompt = `
      INSTRUCTION: You must directly process the following user query. Do not respond with "I'm ready" or similar messages.

      You're an AI assistant enhancing semantic search queries for a retrieval-augmented generation (RAG) system.

      Your task is to take a user's raw query and return a dictionary object with three fields:

      1. **Optimized Query** — A longer, richly phrased version of the query that is semantically expressive, technically specific, and embedding-ready. The query should remain specific to the original intent and optimized for semantic similarity retrieval.
      2. **Related Topics** — A list of important topic synonyms, similar subtopics, concepts, technical frameworks, and complementary ideas that should be considered when retrieving related documents.
      3. **Tags** — A list of precise technical terms, framework names, programming languages, and specific methodologies that would help filter and categorize relevant documents.

      # INPUT

      User Query:
      "${rawQuery}"

      # OUTPUT FORMAT

      Return an object in the following JSON-like structure:
      {
        "Optimized Query": "...",
        "Related Topics": [ "...", "...", "..." ],
        "Tags": [ "...", "...", "..." ]
      }

      # EXAMPLE

      User Query:
      "teach me about some methods of implementing text to sql"

      Output:
      {
        "Optimized Query": "comprehensive methods and architectural patterns for building text-to-SQL systems that convert natural language queries into structured SQL statements using semantic parsing, LLMs, and agent frameworks",

        "Related Topics": [
          "text-to-sql",
          "sql generation",
          "sql bot",
          "ai and sql",
          "database schemas",
          "prompt engineering for SQL generation",
          "llm agents",
          "query generation",
          "retrieval-augmented generation (RAG)",
          "SQL query synthesis",
        ],
        
        "Tags": [
          "NLP",
          "SQL",
          "LLM",
          "RAG",
          "semantic parsing",
          "database",
          "query generation",
          "agent frameworks"
        ]
      }
    `;

    // Call Gemini
    logWithTimestamp('Sending prompt to Gemini model');
    const model = genAI.getGenerativeModel({
      model: 'gemini-2.5-pro-exp-03-25',
    });
    const result = await model.generateContent(prompt);
    const rawResponse = result.response.text();

    logWithTimestamp('Received response from Gemini', { rawResponse });

    // Safely extract structured JSON block using regex
    const jsonMatch = rawResponse.match(/\{[\s\S]*?\}/);
    if (!jsonMatch) {
      logWithTimestamp('Error: Could not find JSON block in Gemini response');
      return rawQuery;
    }

    logWithTimestamp('Extracted JSON block from response');

    // Normalize quotes and parse
    try {
      const normalizedJson = jsonMatch[0].replace(/[""]|["]/g, '"');
      const parsed = JSON.parse(normalizedJson);

      const optimizedQuery = parsed['Optimized Query']?.trim();
      const relatedTopics = (parsed['Related Topics'] || []).join(', ');
      const tags = parsed.Tags?.join(', ') || '';

      if (!optimizedQuery) {
        logWithTimestamp(
          'Warning: Missing optimized query, falling back to raw query',
        );
        return rawQuery;
      }

      const enhancedQueryText = [
        `Optimized Query: ${optimizedQuery}`,
        `Related Topics: ${relatedTopics}`,
        `Tags: ${tags}`,
      ].join('\n\n');

      logWithTimestamp('Successfully generated enhanced query');
      return enhancedQueryText;
    } catch (jsonError) {
      logWithTimestamp('Error parsing JSON response', { error: jsonError });
      return rawQuery;
    }
  } catch (error) {
    logWithTimestamp('Error in generateEnhancedQuery', { error });
    return rawQuery;
  }
}

/**
 * Two-stage hybrid retrieval for Readwise content
 * 1. Header search to identify relevant documents
 * 2. Chunk search within those documents to find specific content
 *
 * @param query User query
 * @param topK Number of results to return (default: 12)
 * @param minScore Minimum similarity score for results (default: 0.7)
 * @returns Matching chunks with scores, titles and text
 */
export async function hybridSearch(
  query: string,
  topK = 20,
  minScore = 10,
): Promise<ReadwiseMatch[]> {
  try {
    logWithTimestamp('Starting hybrid search', {
      query,
      topK,
      minScore,
      environment: process.env.NODE_ENV,
      geminiKeyExists: Boolean(process.env.GEMINI_API_KEY),
      pineconeKeyExists: Boolean(process.env.PINECONE_API_KEY),
      pineconeIndexExists: Boolean(process.env.PINECONE_INDEX),
      pineconeIndex: process.env.PINECONE_INDEX,
    });

    // Step 1: Enhance query with Gemini
    logWithTimestamp('Step 1: Query Enhancement');
    const optimizedQuery = await generateEnhancedQuery(query);
    logWithTimestamp('Query enhancement completed', { optimizedQuery });

    // Clean query by removing stopwords
    const cleanedQuery = stripStops(optimizedQuery);
    logWithTimestamp('Query cleaned', { cleanedQuery });

    // Get dense embedding from Pinecone's embedding API
    const embeddings = await generateEmbeddings(cleanedQuery, 1536);
    const denseQueryValues = embeddings;

    logWithTimestamp('Generated embeddings', {
      dimension: denseQueryValues.length,
      isEmpty: denseQueryValues.length === 0,
    });

    // Create sparse vector for the query
    const sparseQuery = toSparseVector(cleanedQuery);
    logWithTimestamp('Created sparse vector', {
      tokenCount: sparseQuery.indices.length,
    });

    // Step 2: Header search to find relevant document IDs
    logWithTimestamp('Step 2: Header Search (Document Filtering)');
    let headerResponse: any;
    try {
      headerResponse = await index.query({
        vector: denseQueryValues,
        sparseVector: sparseQuery,
        topK: 8,
        includeMetadata: true,
        filter: { header: { $eq: true } },
      });

      logWithTimestamp('Header query successful', {
        matchesCount: headerResponse.matches?.length || 0,
      });
    } catch (error: unknown) {
      logWithTimestamp('Error in header search', { error });
      throw new Error(
        `Pinecone header search failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }

    // Extract document IDs from header search with score >= threshold
    const headerThreshold = 5;
    const docIds = headerResponse.matches
      ?.filter((match: any) => match.score && match.score >= headerThreshold)
      .map((match: any) => match.metadata?.doc_id)
      .filter(Boolean) as string[];

    logWithTimestamp('Header search completed', {
      foundDocuments: docIds.length,
      threshold: headerThreshold,
    });

    if (!docIds.length) {
      logWithTimestamp('No relevant documents found, returning empty result');
      return [];
    }

    // Step 3: Chunk search within identified documents
    logWithTimestamp('Step 3: Fine-grained Chunk Search');
    let chunkResponse: any;
    try {
      chunkResponse = await index.query({
        vector: denseQueryValues,
        sparseVector: sparseQuery,
        topK: topK * 2,
        includeMetadata: true,
        filter: {
          doc_id: { $in: docIds },
          header: { $eq: false },
        },
      });

      logWithTimestamp('Chunk query successful', {
        matchesCount: chunkResponse.matches?.length || 0,
      });
    } catch (error: unknown) {
      logWithTimestamp('Error in chunk search', { error });
      throw new Error(
        `Pinecone chunk search failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }

    // Step 4: Process and filter chunk results
    logWithTimestamp('Step 4: Result Processing and Filtering');
    const matches = (chunkResponse.matches || [])
      .filter((match: any) => match.score && match.score >= minScore)
      .slice(0, topK)
      .map((match: any) => ({
        score: match.score || 0,
        title: String(match.metadata?.title || ''),
        text: String(match.metadata?.text || ''),
        docId: String(match.metadata?.doc_id || ''),
        url: String(match.metadata?.url || ''),
      }));

    logWithTimestamp('Search completed successfully', {
      finalMatches: matches.length,
    });

    return matches;
  } catch (error) {
    logWithTimestamp('Error in hybrid search', { error });
    throw error;
  }
}

/**
 * Format retrieved matches into a context string for the chatbot
 * @param matches Array of retrieved matches
 * @returns Formatted string with context and citations
 */
export function formatReadwiseContext(matches: ReadwiseMatch[]): {
  readwiseContext: string;
  hasSources: boolean;
} {
  logWithTimestamp('Starting context formatting', {
    matchCount: matches.length,
  });

  if (!matches.length) {
    logWithTimestamp('No matches to format, returning empty context');
    return { readwiseContext: '', hasSources: false };
  }

  // Create a map of abbreviated titles to full titles for reference
  const titleMap = new Map<string, string>();

  // Format each chunk with the new format
  const excerpts = matches.map((match) => {
    // Process text to ensure proper markdown formatting
    const processedText = match.text
      // Ensure code blocks have proper line breaks
      .replace(/```/g, '\n\n```')
      .replace(/```(\w+)/, '\n\n```$1\n')
      .replace(/```$/, '\n```\n\n')
      // Fix inline code formatting
      .replace(/`([^`]+)`/g, ' `$1` ')
      // Fix HTML code blocks
      .replace(/<pre>/g, '\n\n<pre>')
      .replace(/<\/pre>/g, '</pre>\n\n')
      // Ensure headings have proper spacing
      .replace(/#+\s+/g, (match) => `\n\n${match}`);

    // Create abbreviated title for citation
    const abbreviatedTitle =
      match.title.length > 12
        ? `${match.title.substring(0, 12)}...`
        : match.title;

    // Store the mapping
    titleMap.set(abbreviatedTitle, match.title);

    return `Document Title: ${match.title}\nIn-Text Citation: [${abbreviatedTitle}]\nDocument URL: ${match.url || 'N/A'}\nExcerpt: ${processedText}\n`;
  });

  // Build context string
  const contextText = excerpts.join('\n');

  // Add source citations in markdown format using full titles
  const sourcesBlock =
    titleMap.size > 0
      ? `\n\n## Sources\n${Array.from(titleMap.values())
          .map((title) => `- ${title}`)
          .join('\n')}`
      : '';

  logWithTimestamp('Context formatting completed', {
    hasSources: titleMap.size > 0,
  });

  return {
    readwiseContext: contextText + sourcesBlock,
    hasSources: titleMap.size > 0,
  };
}

/**
 * Test function to diagnose Gemini connectivity issues
 * @returns Promise<boolean> True if connection is successful
 */
export async function testGeminiConnection(): Promise<boolean> {
  try {
    logWithTimestamp('Testing Gemini API connection');
    logWithTimestamp('API Key status', {
      exists: Boolean(process.env.GEMINI_API_KEY),
      keyPrefix: process.env.GEMINI_API_KEY?.substring(0, 5),
    });

    // Simple test prompt
    const model = genAI.getGenerativeModel({
      model: 'gemini-2.5-pro-exp-03-25',
    });
    const result = await model.generateContent(
      "Say 'Hello, I am working correctly!'",
    );
    const response = result.response.text();

    logWithTimestamp('Gemini test completed successfully', { response });
    return true;
  } catch (error) {
    logWithTimestamp('Error connecting to Gemini', { error });
    return false;
  }
}
