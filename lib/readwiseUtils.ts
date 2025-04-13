import { Pinecone } from '@pinecone-database/pinecone';
import { getEmbedding } from '@/lib/ai/embeddings';

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

const index = pc.index(process.env.PINECONE_INDEX || 'reader-embeddings');

/**
 * Search for Readwise content based on a query
 * @param query - The user query to search for
 * @param topK - Number of results to return (default: 8)
 * @param minScore - Minimum similarity score to include results (default: 0.80)
 * @returns A string containing the relevant Readwise excerpts
 */
export async function getReadwiseContext(
  query: string,
  topK: number = 8,
  minScore: number = 0.80
): Promise<{ readwiseContext: string; hasSources: boolean }> {
  try {
    console.log(`Generating embedding for query: "${query}"`);
    
    // Generate embedding for the query
    const embedding = await getEmbedding(query);
    console.log(`Embedding generated, length: ${embedding.length}`);
    
    // Query Pinecone
    console.log(`Querying Pinecone index: ${process.env.PINECONE_INDEX || 'reader-embeddings'}`);
    console.log(`Parameters: topK=${topK}, minScore=${minScore}`);
    
    const response = await index.query({
      vector: embedding,
      topK,
      includeMetadata: true
    });
    
    console.log(`Pinecone query results: ${response.matches?.length || 0} matches found`);
    
    // Print all matches before filtering
    if (response.matches && response.matches.length > 0) {
      console.log("\n===== ALL PINECONE RESULTS =====");
      response.matches.forEach((match, index) => {
        const metadata = match.metadata || {};
        const title = String(metadata.title || "Untitled");
        const text = String(metadata.text || "");
        const score = match.score || 0;
        
        // Create a preview of the text (first 60 chars)
        const preview = text.length > 60 ? 
          text.substring(0, 60).trim() + "..." : 
          text.trim();
        
        console.log(`[${index + 1}] Score: ${score.toFixed(4)} | Title: "${title}"`);
        console.log(`    Preview: "${preview}"`);
        console.log('');
      });
      console.log("===================================\n");
    }
    
    // Filter results by score
    const matches = (response.matches || [])
      .filter(match => (match.score || 0) >= minScore);
    
    console.log(`Filtered to ${matches.length} matches with score >= ${minScore}`);
    
    if (matches.length === 0) {
      console.log("No matches found after filtering by score");
      return { readwiseContext: "", hasSources: false };
    }
    
    // Log each match with its score
    console.log("\n===== QUERY RESULTS WITH SCORES =====");
    matches.forEach((match, index) => {
      const metadata = match.metadata || {};
      const title = String(metadata.title || "Untitled");
      const text = String(metadata.text || "");
      const score = match.score || 0;
      
      // Create a preview of the text (first 60 chars)
      const preview = text.length > 60 ? 
        text.substring(0, 60).trim() + "..." : 
        text.trim();
      
      console.log(`[${index + 1}] Score: ${score.toFixed(4)} | Title: "${title}"`);
      console.log(`    Preview: "${preview}"`);
      console.log('');
    });
    console.log("=====================================\n");
    
    // Build context string from matches
    const excerpts = matches.map(match => {
      const metadata = match.metadata || {};
      const title = String(metadata.title || "Untitled");
      const text = String(metadata.text || "");
      
      console.log(`Match: "${title}" with score ${match.score}`);
      
      // Ensure code blocks aren't nested in paragraphs by adding line breaks
      const formattedText = text
        .replace(/```/g, '\n\n```')
        .replace(/`/g, ' ` ')
        .replace(/<pre>/g, '\n\n<pre>')
        .replace(/<\/pre>/g, '</pre>\n\n');
      
      return `### ${title}\n\n${formattedText}`;
    });
    
    const readwiseContext = excerpts.join("\n\n");
    
    // Extract source titles for citation
    const sources = [...new Set(
      matches
        .map(match => match.metadata?.title)
        .filter(Boolean) as string[]
    )];
    
    console.log(`Sources found: ${sources.length}`);
    sources.forEach(source => console.log(`- ${source}`));
    
    const sourcesBlock = sources.length > 0 
      ? "\n\nSources:\n" + sources.map(title => `- ${title}`).join("\n")
      : "";
    
    return { 
      readwiseContext: sources.length > 0 ? readwiseContext + sourcesBlock : "",
      hasSources: sources.length > 0
    };
  } catch (error) {
    console.error("Error fetching Readwise context:", error);
    if (error instanceof Error) {
      console.error(`Error name: ${error.name}`);
      console.error(`Error message: ${error.message}`);
      console.error(`Error stack: ${error.stack}`);
    }
    return { readwiseContext: "", hasSources: false };
  }
} 