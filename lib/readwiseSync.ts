import { Pinecone } from '@pinecone-database/pinecone';
// Replace tiktoken with a simpler tokenization approach
import 'dotenv/config';

// Update to use the correct Reader API endpoint
const RW_API = 'https://readwise.io/api/v3/list/';
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});
// Update to use reader-embeddings index
const index = pc.index(process.env.PINECONE_INDEX || 'reader-embeddings');

// Function to fetch all existing document IDs from Pinecone
async function fetchExistingDocumentIds(): Promise<Set<string>> {
  const existingIds = new Set<string>();
  
  try {
    console.log("Fetching existing document IDs from Pinecone...");
    
    // Get index stats to understand the size of the database
    const stats = await index.describeIndexStats();
    const totalVectors = stats.totalRecordCount || 0;
    
    console.log(`Pinecone index contains ${totalVectors} total vectors`);
    
    if (totalVectors === 0) {
      console.log("Index is empty, no existing documents to check");
      return existingIds;
    }
    
    // Since Pinecone requires filters or vectors, we'll use a zero vector approach
    console.log("Using zero vector query to fetch document IDs");
    
    try {
      // Create a zero vector with the same dimension as your index (1024, not 4096)
      const zeroVector = Array(1024).fill(0);
      
      // Query with the zero vector to get random results across the index
      const response = await index.query({
        vector: zeroVector,
        topK: 10000,
        includeMetadata: true
      });
      
      const matches = response.matches || [];
      console.log(`Retrieved ${matches.length} vectors with zero vector query`);
      
      // Extract document IDs
      for (const match of matches) {
        if (match.metadata && match.metadata.doc_id) {
          existingIds.add(match.metadata.doc_id as string);
        } else if (match.id) {
          // Fallback to parsing the vector ID
          const docId = match.id.split('-')[0];
          if (docId) {
            existingIds.add(docId);
          }
        }
      }
      
      console.log(`Found ${existingIds.size} unique document IDs`);
      
    } catch (error) {
      console.error("Error fetching document IDs:", error);
      console.log("Will proceed without document deduplication");
    }
    
    return existingIds;
    
  } catch (error) {
    console.error("Error in fetchExistingDocumentIds:", error);
    // Return empty set if there's an error
    return new Set<string>();
  }
}

// TypeScript type for Reader API response
type ReaderDoc = {
  id: string;
  title: string;
  author: string;
  url: string;
  category: string;      // pdf | article | tweet | ...
  html_content?: string | null;  // cleaned Reader HTML is in this field
  content?: string | null;
  summary?: string | null;
  word_count: number;
  created_at: string;
  updated_at: string;
  source_url?: string;
};

// Function to extract text from HTML
function extractTextFromHtml(html: string): string {
  // Simple HTML to text conversion (replace with more robust solution if needed)
  return html
    .replace(/<[^>]*>/g, ' ')  // Strip HTML tags
    .replace(/&nbsp;/g, ' ')   // Replace non-breaking spaces
    .replace(/&amp;/g, '&')    // Replace ampersands
    .replace(/&lt;/g, '<')     // Replace less than
    .replace(/&gt;/g, '>')     // Replace greater than
    .replace(/&quot;/g, '"')   // Replace quotes
    .replace(/&#39;/g, "'")    // Replace apostrophes
    .replace(/\s+/g, ' ')      // Collapse whitespace
    .trim();                   // Trim leading/trailing whitespace
}

// Simple function to count tokens (approximate)
function countTokens(text: string): number {
  return Math.ceil(text.split(/\s+/).length * 1.3);
}

/**
 * Split a document into ≤ maxTokens chunks.
 * – First split by blank‑line paragraphs.
 * – Merge paragraphs until we'd exceed maxTokens.
 * – Add sentence‑level overlap (carry‑over) on both sides to preserve context.
 *
 * @param text          Cleaned plain‑text document
 * @param maxTokens     Hard token ceiling for each chunk
 * @param paraOverlap   # sentences of the *previous* paragraph to prepend
 * @param nextOverlap   # sentences of the *next* paragraph to append
 */
function semanticChunksWithOverlap(
  text: string,
  maxTokens = 512,
  paraOverlap = 1,
  nextOverlap = 1,
): string[] {
  const paragraphs = text
    .split(/\n{2,}/)           // blank line ⇒ new paragraph
    .map(p => p.trim())
    .filter(Boolean);

  const out: string[] = [];

  // Caching token counts
  const paraTokenCounts = new Map<string, number>();
  const getTokens = (s: string) => {
    if (!paraTokenCounts.has(s)) paraTokenCounts.set(s, countTokens(s));
    return paraTokenCounts.get(s)!;
  };

  // Fallback-safe sentence splitter
  const splitSentences = (p: string): string[] => {
    const match = p.match(/[^.!?]+[.!?]+["')\]]*|\S+$/g);
    return match && match.length > 0 ? match : [p];
  };

  let buf = "";
  let i = 0;

  while (i < paragraphs.length) {
    const para = paragraphs[i];
    const candidate = [buf, para].filter(Boolean).join("\n\n");

    if (!buf && getTokens(para) > maxTokens) {
      // Paragraph alone too big → split by sentence
      const sentences = splitSentences(para);
      let chunk = "";
      for (const sentence of sentences) {
        const next = chunk ? chunk + " " + sentence : sentence;
        if (getTokens(next) > maxTokens) {
          if (chunk) out.push(chunk.trim());
          chunk = sentence;
        } else {
          chunk = next;
        }
      }
      if (chunk) out.push(chunk.trim());
      i++;
      continue;
    }

    if (getTokens(candidate) > maxTokens) {
      const nextPara = paragraphs[i] || "";
      const nextSentences = splitSentences(nextPara).slice(0, nextOverlap);
      const overlapText = nextSentences.length ? "\n\n" + nextSentences.join(" ") : "";
      out.push(buf.trim() + overlapText);
      buf = ""; // reset, but don’t increment i
    } else {
      buf = candidate;
      i++; // consume this paragraph
    }
  }

  // Push final buffer, with tail overlap from previous chunk
  if (buf) {
    const lastChunkIdx = out.length - 1;
    if (lastChunkIdx >= 0) {
      const prevSentences = splitSentences(out[lastChunkIdx]).slice(-paraOverlap);
      buf = prevSentences.join(" ") + "\n\n" + buf;
    }
    out.push(buf.trim());
  }

  return out.map(c => c.trim());
}


// Simple function to chunk text by estimated token count
function chunkText(text: string, maxTokens = 512): string[] {
  const chunks: string[] = [];
  const words = text.split(/\s+/);
  let currentChunk: string[] = [];
  let currentTokenCount = 0;
  
  for (const word of words) {
    // Estimate tokens for this word (including space)
    const wordTokens = Math.ceil((word.length + 1) * 0.3);
    
    if (currentTokenCount + wordTokens > maxTokens && currentChunk.length > 0) {
      chunks.push(currentChunk.join(' '));
      currentChunk = [word];
      currentTokenCount = wordTokens;
    } else {
      currentChunk.push(word);
      currentTokenCount += wordTokens;
    }
  }
  
  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(' '));
  }
  
  return chunks;
}

export async function syncReadwise(updatedAfter?: string, forceUpdate = false): Promise<void> {
  // Track processed documents to avoid duplicates across all locations
  const processedDocIds = new Set<string>();
  // Define locations to fetch
  const locations = ["new", "later", "archive"];
  
  try {
    console.log("Starting Readwise Reader sync process...");
    console.log(`API Token exists: ${Boolean(process.env.READWISE_TOKEN)}`);
    
    // Only fetch existing document IDs if we're not forcing an update
    let existingDocIds = new Set<string>();
    if (!forceUpdate) {
      existingDocIds = await fetchExistingDocumentIds();
      console.log(`Will skip ${existingDocIds.size} documents that already exist in Pinecone`);
    } else {
      console.log("Force update mode: Will process all documents regardless of existing vectors");
    }
    
    // Track stats across all locations
    let totalDocumentsReceived = 0;
    let documentsSkipped = 0;
    let documentsProcessed = 0;
    let chunksEmbedded = 0;
    
    // Process each location separately
    for (const location of locations) {
      console.log(`\n=== PROCESSING LOCATION: ${location.toUpperCase()} ===\n`);
      
      // Process each location with its own pagination
      let cursor: string | null = null;
      
      while (true) {
        // URL with withHtmlContent=true flag and location filter
        let url = `${RW_API}?withHtmlContent=true&location=${location}`;
        
        if (cursor) {
          url += `&pageCursor=${cursor}`;
        }
        
        if (updatedAfter && !cursor) {
          url += `&updatedAfter=${updatedAfter}`;
        }

        console.log(`Fetching from Readwise Reader API (${location}): ${url.replace(/Token [^&]+/, 'Token [REDACTED]')}`);
        
        const res = await fetch(url, {
          headers: { 
            Authorization: `Token ${process.env.READWISE_TOKEN}`,
            Accept: 'application/json'
          },
        });
        
        console.log(`Response status: ${res.status} ${res.statusText}`);
        
        if (res.status === 429) {
          const wait = +(res.headers.get('Retry-After') || '60') * 1000;
          console.log(`Rate limited, waiting ${wait}ms before retrying`);
          await new Promise(r => setTimeout(r, wait));
          continue;
        }
        
        if (!res.ok) {
          const responseText = await res.text();
          console.error(`Error response from Readwise API: ${responseText}`);
          throw new Error(`HTTP error ${res.status}: ${responseText}`);
        }
        
        // Check if response is JSON
        const contentType = res.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
          const text = await res.text();
          console.error(`Received non-JSON response: ${text}`);
          throw new Error('Received non-JSON response from Readwise API');
        }
        
        let data;
        try {
          data = await res.json() as { 
            results: ReaderDoc[],
            nextPageCursor?: string 
          };
        } catch (error) {
          const text = await res.text();
          console.error(`Failed to parse JSON: ${error}`);
          console.error(`Response text: ${text}`);
          throw error;
        }
        
        const { results, nextPageCursor } = data;
        const batchSize = results?.length || 0;
        totalDocumentsReceived += batchSize;
        console.log(`Received ${batchSize} documents from Readwise Reader for ${location} (total across locations: ${totalDocumentsReceived})`);

        // Process each document
        for (const doc of results) {
          // Skip if already processed in this run to avoid duplicates
          if (processedDocIds.has(doc.id)) {
            console.log(`Skipping duplicate document in this batch: ${doc.title} (${doc.id})`);
            documentsSkipped++;
            continue;
          }
          processedDocIds.add(doc.id);
          
          // Skip if document already exists in Pinecone and we're not forcing updates
          if (!forceUpdate && existingDocIds.has(doc.id)) {
            console.log(`Skipping existing document: ${doc.title} (${doc.id})`);
            documentsSkipped++;
            continue;
          }
          
          console.log(`Processing document: ${doc.title || 'Untitled'} (${doc.category})`);
          
          // Extract text content from the document
          let textContent = '';
          
          if (doc.html_content && doc.html_content.trim()) {
            textContent = extractTextFromHtml(doc.html_content);
            console.log(`Using HTML content (${doc.html_content.length} chars)`);
          } else if (doc.content && doc.content.trim()) {
            textContent = extractTextFromHtml(doc.content);
            console.log(`Using content field (${doc.content.length} chars)`);
          } else if (doc.summary && doc.summary.trim()) {
            // Some PDFs come back with plain-text summary
            textContent = doc.summary.trim();
            console.log(`Using summary field (${doc.summary.length} chars)`);
          } else {
            console.log(`Document has no parsable content, skipping: ${doc.title}`);
            console.log('Available fields:', Object.keys(doc).filter(key => doc[key as keyof ReaderDoc]));
            documentsSkipped++;
            continue;
          }
          
          console.log(`Extracted ${textContent.length} characters of text`);
          
          // Skip if no text was extracted
          if (!textContent || textContent.trim() === '') {
            console.log(`No text could be extracted from document: ${doc.title}`);
            documentsSkipped++;
            continue;
          }
          
          // Collect all chunks for this document
          const documentChunks: { id: string, text: string, metadata: any }[] = [];
          
          // Chunk text
          try {
            const chunks = semanticChunksWithOverlap(textContent, 512, 1, 1);
            console.log(`Split document into ${chunks.length} chunks`);
            
            for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
              const chunkText = chunks[chunkIndex];
              
              documentChunks.push({
                id: `${doc.id}-${chunkIndex}`,
                text: chunkText,
                metadata: {
                  doc_id: doc.id,
                  title: doc.title,
                  author: doc.author,
                  source_url: doc.source_url || doc.url,
                  category: doc.category,
                  location: location, // Add location to metadata
                  url: doc.url,
                  created_at: doc.created_at,
                  updated_at: doc.updated_at,
                  chunk_index: chunkIndex,
                  total_chunks: chunks.length,
                  text: chunkText,
                }
              });
            }
          } catch (error) {
            console.error("Error processing document:", error);
            documentsSkipped++;
            continue;
          }
          
          // Process in batches of 50 to avoid overloading the API
          const BATCH_SIZE = 50;
          for (let i = 0; i < documentChunks.length; i += BATCH_SIZE) {
            const batch = documentChunks.slice(i, i + BATCH_SIZE);
            
            if (batch.length === 0) continue;
            
            try {
              console.log(`Embedding batch of ${batch.length} chunks using Pinecone...`);
              
              // Generate embeddings using Pinecone's embedding service
              const embeddingResponse = await pc.inference.embed(
                "llama-text-embed-v2",
                batch.map(item => item.text),
                {
                  input_type: "passage"
                }
              );
              
              // Log embedding info but not the actual vectors (too large)
              console.log("\nEMBEDDING INFO:");
              console.log(`Model used: ${embeddingResponse.model}`);
              console.log(`Vector type: ${embeddingResponse.vectorType}`);
              console.log(`Number of embeddings: ${embeddingResponse.data?.length || 0}`);
              console.log(`Total tokens: ${embeddingResponse.usage?.totalTokens || 'unknown'}`);
              
              // Check that we have a valid embedding response
              if (!embeddingResponse || !embeddingResponse.data || !Array.isArray(embeddingResponse.data)) {
                console.error("Unexpected embedding response format:", embeddingResponse);
                throw new Error("Invalid embedding response format");
              }
              
              // Prepare vectors for upsert - use type assertion to handle the embedding structure
              const vectors = batch.map((item, idx) => {
                const embeddingData = embeddingResponse.data[idx] as any; // Type assertion to bypass TS error
                return {
                  id: item.id,
                  values: embeddingData.values,
                  metadata: item.metadata
                };
              });
              
              // Upsert the vectors
              console.log(`Upserting ${vectors.length} vectors to Pinecone index`);
              await index.upsert(vectors);
              chunksEmbedded += vectors.length;
              
            } catch (error) {
              console.error("Error generating embeddings or upserting vectors:", error);
              // We don't skip the document on embedding error since some chunks might succeed
            }
          }
          
          documentsProcessed++;
        }

        if (!nextPageCursor) {
          console.log(`No more pages for location "${location}", moving to next location`);
          break;
        }
        cursor = nextPageCursor;
        console.log(`Moving to next page for location "${location}" with cursor: ${cursor}`);
      }
      
      // Location completion status
      console.log(`\n=== Completed location: ${location} ===\n`);
    }
    
    // Log final stats
    console.log("\n=== SYNC SUMMARY ===");
    console.log(`Total documents received: ${totalDocumentsReceived}`);
    console.log(`Documents processed: ${documentsProcessed}`);
    console.log(`Documents skipped: ${documentsSkipped}`);
    console.log(`Chunks embedded: ${chunksEmbedded}`);
    console.log(`Locations processed: ${locations.join(', ')}`);
    console.log("====================");
    
  } catch (error) {
    console.error("Error in syncReadwise:", error);
    throw error;
  }
}