import { Pinecone } from '@pinecone-database/pinecone';
import { stripStops } from './stopwords';
import { toSparseVector } from './sparse';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Initialize Google Generative AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// Embedding model
const EMBED_MODEL = 'llama-text-embed-v2';

// Initialize Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

// Get index
const index = pc.index(process.env.PINECONE_INDEX || 'reader-embeddings');

interface ReadwiseMatch {
  score: number;
  title: string;
  text: string;
  docId?: string;
}

/**
 * Use Gemini to enhance a search query for better semantic matching
 * @param rawQuery User's original query
 * @returns Enhanced version of the query optimized for vector search
 */
async function generateEnhancedQuery(rawQuery: string): Promise<string> {
  try {
    console.log("Calling Gemini to enhance query...");
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
    console.log("Sending prompt to Gemini model...");
    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });
    const result = await model.generateContent(prompt);
    const rawResponse = result.response.text();
    
    console.log("=== RAW GEMINI RESPONSE ===");
    console.log(rawResponse);
    console.log("===========================");

    // Safely extract structured JSON block using regex
    const jsonMatch = rawResponse.match(/\{[\s\S]*?\}/);
    if (!jsonMatch) {
      console.error("Could not find JSON block in Gemini response.");
      return rawQuery;
    }
    
    console.log("=== EXTRACTED JSON BLOCK ===");
    console.log(jsonMatch[0]);
    console.log("============================");

    // Normalize quotes and parse
    try {
      const normalizedJson = jsonMatch[0].replace(/[""]|["]/g, '"');
      console.log("=== NORMALIZED JSON ===");
      console.log(normalizedJson);
      console.log("=======================");
      
      const parsed = JSON.parse(normalizedJson);
      
      console.log("=== PARSED JSON ===");
      console.log(JSON.stringify(parsed, null, 2));
      console.log("===================");

      const optimizedQuery = parsed["Optimized Query"]?.trim();
      const relatedTopics = (parsed["Related Topics"] || []).join(", ");
      const tags = (parsed["Tags"] || []).join(", ");

      if (!optimizedQuery) {
        console.warn("Missing optimized query. Falling back to raw query.");
        return rawQuery;
      }

      const enhancedQueryText = [
        `Optimized Query: ${optimizedQuery}`,
        `Related Topics: ${relatedTopics}`,
        `Tags: ${tags}`
      ].join("\n\n");
      
      console.log("=== FINAL ENHANCED QUERY ===");
      console.log(enhancedQueryText);
      console.log("============================");
      
      return enhancedQueryText;
    } catch (jsonError) {
      console.error("JSON parsing error:", jsonError);
      console.error("Failed to parse JSON:", jsonMatch[0]);
      return rawQuery;
    }
  } catch (error) {
    console.error("Error generating enhanced query with Gemini:", error);
    if (error instanceof Error) {
      console.error("Error details:", error.message);
      console.error("Error stack:", error.stack);
    }
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
  topK: number = 12,
  minScore: number = 0.7
): Promise<ReadwiseMatch[]> {
  try {
    console.log(`Hybrid search for query: "${query}"`);
    
    // Log search parameters
    console.log("\n===== SEARCH PARAMETERS =====");
    console.log(`Minimum score threshold for chunks: ${minScore}`);
    console.log(`Minimum score threshold for headers: 10.0`);
    console.log(`Maximum results to return: ${topK}`);
    console.log(`Index: ${process.env.PINECONE_INDEX || 'reader-embeddings'}`);
    console.log(`Embedding model: ${EMBED_MODEL}`);
    console.log("=============================\n");
    
    // Step 1: Enhance query with Gemini
    console.log("STEP 1: Query Enhancement");
    const optimizedQuery = await generateEnhancedQuery(query);
    console.log(`Gemini-enhanced query: "${optimizedQuery}"`);
    
    // Clean query by removing stopwords
    const cleanedQuery = stripStops(optimizedQuery);
    console.log(`Cleaned query: "${cleanedQuery}"`);
    
    // Get dense embedding from Pinecone's embedding API
    console.log(`Generating dense embedding with ${EMBED_MODEL}`);
    const { data: embeddings } = await pc.inference.embed(
      EMBED_MODEL,
      [cleanedQuery],
      { input_type: 'query' }
    );
    
    // Extract vector values from embedding
    const denseQueryValues = Array.isArray(embeddings[0]) 
      ? embeddings[0] 
      : 'values' in embeddings[0]
        ? (embeddings[0] as any).values
        : [];
        
    console.log(`Dense embedding created, dimension: ${denseQueryValues.length}`);
    
    // Create sparse vector for the query
    const sparseQuery = toSparseVector(cleanedQuery);
    console.log(`Sparse vector created with ${sparseQuery.indices.length} tokens`);
    
    // Step 2: Header search to find relevant document IDs
    console.log("\nSTEP 2: Header Search (Document Filtering)");
    const headerResponse = await index.query({
      vector: denseQueryValues,
      sparseVector: sparseQuery,
      topK: 20,
      includeMetadata: true,
      filter: { header: { $eq: true } }
    });
    
    // Extract document IDs from header search with score >= threshold
    const headerThreshold = 1.0;
    const docIds = headerResponse.matches
      ?.filter(match => match.score && match.score >= headerThreshold)
      .map(match => match.metadata?.doc_id)
      .filter(Boolean) as string[];
    
    console.log(`Found ${docIds.length} relevant document IDs with score >= ${headerThreshold}`);
    
    // Log the header matches with their scores for debugging
    console.log("\n===== ALL DOCUMENT HEADERS WITH SCORES =====");
    headerResponse.matches?.forEach((match, idx) => {
      const score = match.score || 0;
      const isRelevant = score >= headerThreshold;
      const docId = match.metadata?.doc_id || 'unknown';
      const title = match.metadata?.title || 'Untitled';
      
      console.log(
        `[${idx+1}] ${isRelevant ? '✓' : '✗'} Score: ${score.toFixed(4)} | ` +
        `Doc ID: ${docId} | Title: "${title}"`
      );
    });
    console.log("===========================================\n");
    
    if (!docIds.length) {
      console.log(`No relevant documents found, returning empty result`);
      return [];
    }
    
    // Step 3: Chunk search within identified documents
    console.log("\nSTEP 3: Fine-grained Chunk Search");
    const chunkResponse = await index.query({
      vector: denseQueryValues,
      sparseVector: sparseQuery,
      topK: topK * 2, // Request more to allow for filtering
      includeMetadata: true,
      filter: { 
        doc_id: { $in: docIds },
        header: { $eq: false }
      }
    });
    
    // Log all chunks with their scores
    console.log("\n===== ALL CHUNKS WITH SCORES =====");
    chunkResponse.matches?.forEach((match, idx) => {
      const score = match.score || 0;
      const isSelected = score >= minScore && idx < topK;
      const title = match.metadata?.title || 'Untitled';
      const docId = match.metadata?.doc_id || 'unknown';
      const preview = match.metadata?.text 
        ? String(match.metadata.text).substring(0, 50).trim() + "..." 
        : "No text";
      
      console.log(
        `[${idx+1}] ${isSelected ? '✓' : '✗'} Score: ${score.toFixed(4)} | ` +
        `Doc ID: ${docId} | Title: "${title}"`
      );
      console.log(`    Preview: "${preview}"`);
    });
    console.log("==================================\n");
    
    // Step 4: Process and filter chunk results
    console.log("\nSTEP 4: Result Processing and Filtering");
    const matches = (chunkResponse.matches || [])
      .filter(match => match.score && match.score >= minScore)
      .slice(0, topK) // Limit to requested number of results
      .map(match => ({
        score: match.score || 0,
        title: String(match.metadata?.title || ""),
        text: String(match.metadata?.text || ""),
        docId: String(match.metadata?.doc_id || "")
      }));
    
    console.log(`Found ${matches.length} matching chunks after filtering`);
    
    // Log the final selected chunks with scores
    console.log("\n===== SELECTED CHUNKS =====");
    matches.forEach((match, i) => {
      console.log(`[${i+1}] Score: ${match.score.toFixed(4)} | Title: "${match.title}"`);
      console.log(`    Preview: "${match.text.substring(0, 100).trim()}..."`);
      console.log('');
    });
    console.log("===========================\n");
    
    return matches;
    
  } catch (error) {
    console.error("Error in hybrid search:", error);
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
  if (!matches.length) {
    return { readwiseContext: "", hasSources: false };
  }
  
  // Format each chunk with proper markdown
  const excerpts = matches.map(match => {
    // Process text to ensure proper markdown formatting
    // Especially for code blocks to avoid nesting issues
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
      .replace(/#+\s+/g, match => `\n\n${match}`);
    
    return `### ${match.title}\n\n${processedText}`;
  });
  
  // Build context string
  const contextText = excerpts.join("\n\n");
  
  // Extract unique source titles for citation
  const sources = [...new Set(matches.map(match => match.title))];
  
  // Add source citations in markdown format
  const sourcesBlock = sources.length > 0 
    ? "\n\n## Sources\n" + sources.map(title => `- ${title}`).join("\n")
    : "";
  
  return {
    readwiseContext: contextText + sourcesBlock,
    hasSources: sources.length > 0
  };
}

/**
 * Test function to diagnose Gemini connectivity issues
 * @returns Promise<boolean> True if connection is successful
 */
export async function testGeminiConnection(): Promise<boolean> {
  try {
    console.log("Testing Gemini API connection...");
    console.log(`API Key exists: ${Boolean(process.env.GEMINI_API_KEY)}`);
    console.log(`API Key starts with: ${process.env.GEMINI_API_KEY?.substring(0, 5)}...`);
    
    // Simple test prompt
    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });
    const result = await model.generateContent("Say 'Hello, I am working correctly!'");
    const response = result.response.text();
    
    console.log("=== GEMINI TEST RESPONSE ===");
    console.log(response);
    console.log("===========================");
    
    return true;
  } catch (error) {
    console.error("Error connecting to Gemini:", error);
    if (error instanceof Error) {
      console.error("Error details:", error.message);
      console.error("Error stack:", error.stack);
    }
    return false;
  }
} 