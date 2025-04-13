import { Pinecone } from '@pinecone-database/pinecone';
// Replace tiktoken with a simpler tokenization approach
import 'dotenv/config';
import { chunkTextWithOverlap } from './ai/chunker';
import { toSparseVector } from './ai/sparse';
import { stripStops } from './ai/stopwords';
import { 
  extractRakeKeywords, 
  extractTfIdfKeywords, 
  addDocumentToTfIdf, 
  buildTfIdfModel,
  boostTerm,
  extractKeywords
} from './ai/extract-keywords';

// Update to use the correct Reader API endpoint
const RW_API = 'https://readwise.io/api/v3/list/';
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});
// Update to use reader-embeddings index
const index = pc.index(process.env.PINECONE_INDEX || 'reader-embeddings');

// Embedding model
const EMBED_MODEL = 'llama-text-embed-v2';

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
  tags?: string[];
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
      buf = ""; // reset, but don't increment i
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

async function processDocuments(docs: ReaderDoc[], existingIds: Set<string>): Promise<void> {
  // First phase: collect all documents for TF-IDF corpus
  console.log("Phase 1: Building TF-IDF corpus...");
  for (const doc of docs) {
    if (existingIds.has(doc.id)) continue;
    
    // Extract text for TF-IDF processing
    let text = "";
    if (doc.html_content) {
      text = extractTextFromHtml(doc.html_content);
    } else if (doc.content) {
      text = doc.content;
    } else {
      continue; // Skip docs with no content
    }
    
    // Add to TF-IDF corpus
    addDocumentToTfIdf(doc.id, text);
  }
  
  // Build the TF-IDF model
  buildTfIdfModel();
  
  // Second phase: process each document with the built model
  console.log("Phase 2: Creating Super-Headers and processing documents...");
  for (const doc of docs) {
    await processDocument(doc, existingIds);
  }
}

async function processDocument(doc: ReaderDoc, existingIds: Set<string>): Promise<void> {
  // Skip if already processed and not forcing update
  if (existingIds.has(doc.id)) {
    console.log(`Document ${doc.id} (${doc.title}) already exists, skipping`);
    return;
  }
  
  console.log(`Processing document: ${doc.title}`);
  
  // Extract text from HTML content or use plain content
  let text = "";
  if (doc.html_content) {
    text = extractTextFromHtml(doc.html_content);
  } else if (doc.content) {
    text = doc.content;
  } else {
    console.log(`No content for ${doc.id}, skipping`);
    return;
  }
  
  // -------------------- SUPER-HEADER CREATION --------------------
  
  // 1. Basic components - title, author, summary
  const title = doc.title || "";
  const author = doc.author || "";
  const summary = doc.summary || "";
  
  console.log(`Creating Super-Header for "${doc.title}"`);
  
  // 2. Get tags (if available)
  const tags = doc.tags || [];
  
  // Debug tags
  console.log('Tags before processing:', JSON.stringify(tags));
  
  // Extract tag strings - handle both string arrays and complex objects
  let tagStrings: string[] = [];
  if (Array.isArray(tags)) {
    tagStrings = tags.map(tag => {
      if (typeof tag === 'string') return tag;
      if (typeof tag === 'object' && tag !== null) {
        // If tag is an object, try to get a string representation
        if ('name' in tag) return String((tag as any).name);
        if ('id' in tag) return String((tag as any).id);
        return String(Object.values(tag)[0] || '');
      }
      return String(tag);
    });
  }
  
  console.log('Tags after processing:', tagStrings);
  const tagsText = tagStrings.length > 0 ? `Tags: ${tagStrings.join(' ')}` : '';
  
  if (tagStrings.length > 0) {
    console.log(`Document has ${tagStrings.length} tags: ${tagStrings.join(', ')}`);
  }
  
  // 3. Extract RAKE keywords and TF-IDF terms
  const { rakeKeywords, tfidfKeywords, boostedText } = extractKeywords(doc.id, text);
  
  console.log(`Extracted ${rakeKeywords.length} RAKE keywords and ${tfidfKeywords.length} TF-IDF terms`);
  
  // Log some of the top terms
  if (rakeKeywords.length > 0) {
    console.log(`Top RAKE terms: ${rakeKeywords.slice(0, 5).map(k => k.term).join(', ')}`);
  }
  
  if (tfidfKeywords.length > 0) {
    console.log(`Top TF-IDF terms: ${tfidfKeywords.slice(0, 5).map(k => k.term).join(', ')}`);
  }
  
  // 🔹 Step 1: Extract relevant fields (already done above)
  
  // 🔹 Step 2: Deduplicate and union key terms
  const rakeTermsSet = new Set(rakeKeywords.map(k => k.term));
  const tfidfTermsSet = new Set(tfidfKeywords.map(k => k.term));
  
  const dedupedRakeTerms = [...rakeTermsSet];
  const dedupedTfidfTerms = [...tfidfTermsSet].filter(term => !rakeTermsSet.has(term));
  
  console.log(`After deduplication: ${dedupedRakeTerms.length} RAKE terms, ${dedupedTfidfTerms.length} unique TF-IDF terms`);
  
  // 🔹 Step 3: Format for dense embedding
  const headerContent = [
    `Title: ${title}`,
    `Author: ${author}`,
    tagsText ? `Tags: ${tagStrings.join(', ')}` : '',
    summary ? `Summary: ${summary}` : '',
    `RAKE Keywords: ${dedupedRakeTerms.join(', ')}`,
    `TF-IDF Top Terms: ${dedupedTfidfTerms.join(', ')}`
  ].filter(Boolean).join('\n\n');
  
  // Replace old superHeaderText with new structured format
  const superHeaderText = headerContent;
  
  // Log the full header content
  console.log(`Super-Header created (${superHeaderText.length} chars)`);
  console.log('======== FULL HEADER CONTENT ========');
  console.log(superHeaderText);
  console.log('=====================================');
  
  // Create header embedding with the super-header text
  const { data: headerEmbeddings } = await pc.inference.embed(
    EMBED_MODEL,
    [superHeaderText],
    { input_type: 'passage' }
  );
  
  // Extract vector values from embedding
  const headerVector = Array.isArray(headerEmbeddings[0]) 
    ? headerEmbeddings[0] 
    : 'values' in headerEmbeddings[0]
      ? (headerEmbeddings[0] as any).values
      : [];
  
  // 🔹 Step 4: Build sparse vector from keywords with weights
  // Create a sparse vector with proper weighting instead of term repetition
  const sparseTermsMap = new Map<string, number>();
  
  // Add RAKE terms with their weights
  for (const { term, weight } of rakeKeywords) {
    sparseTermsMap.set(term, (sparseTermsMap.get(term) || 0) + (weight * 1.5)); // Boost RAKE terms
  }
  
  // Add TF-IDF terms with their weights
  for (const { term, weight } of tfidfKeywords) {
    sparseTermsMap.set(term, (sparseTermsMap.get(term) || 0) + weight);
  }
  
  // Create a weighted sparse vector
  const sparseVector = toSparseVector(
    [...sparseTermsMap.entries()]
      .map(([term, weight]) => ({ term, weight }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 100) // Limit to top 100 terms
      .map(({ term }) => term)
      .join(' ')
  );
  
  // Enhance sparse vector with weighted terms
  const enhancedSparseVector = enhanceSparseVector(
    sparseVector,
    [...rakeKeywords, ...tfidfKeywords]
  );
  
  // 🔹 Step 5: Store in hybrid index
  // Upsert header vector
  await index.upsert([{
    id: `${doc.id}-header`,
    values: headerVector,
    sparseValues: enhancedSparseVector,
    metadata: {
      doc_id: doc.id,
      title: doc.title,
      author: doc.author,
      url: doc.url,
      category: doc.category,
      text: superHeaderText.slice(0, 1000),
      summary: doc.summary || "",
      tags: tagStrings,
      header: true,
      created_at: doc.created_at,
    }
  }]);
  
  // Create meaningful chunks with overlap
  const chunks = chunkTextWithOverlap(text, 512, 2);
  console.log(`Created ${chunks.length} chunks from document`);
  
  // Limit to 20 chunks per document to avoid rate limits
  const chunkLimit = Math.min(chunks.length, 20);
  
  // Process chunks in batches
  for (let i = 0; i < chunkLimit; i++) {
    const chunkText = chunks[i];
    console.log(`Processing chunk ${i+1}/${chunkLimit} (${chunkText.length} chars)`);
    
    // Create dense vector
    const { data: chunkEmbeddings } = await pc.inference.embed(
      EMBED_MODEL,
      [chunkText],
      { input_type: 'passage' }
    );
    
    // Extract vector values
    const chunkVector = Array.isArray(chunkEmbeddings[0]) 
      ? chunkEmbeddings[0] 
      : 'values' in chunkEmbeddings[0]
        ? (chunkEmbeddings[0] as any).values
        : [];
    
    // Create sparse vector
    const chunkSparse = toSparseVector(chunkText);
    
    // Upsert chunk vector
    await index.upsert([{
      id: `${doc.id}-chunk-${i}`,
      values: chunkVector,
      sparseValues: chunkSparse,
      metadata: {
        doc_id: doc.id,
        title: doc.title,
        author: doc.author,
        url: doc.url,
        category: doc.category,
        text: chunkText,
        header: false,
        chunk_id: i,
        created_at: doc.created_at,
      }
    }]);
    
    // Add a small delay to avoid rate limits
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

/**
 * Enhance a sparse vector with term weights
 * @param baseVector Basic sparse vector
 * @param weightedTerms Array of terms with weights
 * @returns Enhanced sparse vector
 */
function enhanceSparseVector(
  baseVector: { indices: number[], values: number[] },
  weightedTerms: Array<{term: string, weight: number}>
): { indices: number[], values: number[] } {
  // Create a map of term hashes to weights
  const termWeights = new Map<number, number>();
  
  // Hash function should match the one in toSparseVector
  const hashFn = (term: string): number => {
    let hash = 0;
    for (let i = 0; i < term.length; i++) {
      hash = ((hash << 5) - hash) + term.charCodeAt(i);
      hash |= 0; // Convert to 32bit integer
    }
    return Math.abs(hash);
  };
  
  // Add weighted terms to the map
  for (const { term, weight } of weightedTerms) {
    const hash = hashFn(term);
    termWeights.set(hash, weight);
  }
  
  // Create a copy of the base vector
  const enhancedVector = {
    indices: [...baseVector.indices],
    values: [...baseVector.values]
  };
  
  // Apply weights to matching indices
  for (let i = 0; i < enhancedVector.indices.length; i++) {
    const index = enhancedVector.indices[i];
    if (termWeights.has(index)) {
      // Boost the weight
      enhancedVector.values[i] *= termWeights.get(index) || 1;
    }
  }
  
  return enhancedVector;
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
    
    // Collect all documents first
    const allDocuments: ReaderDoc[] = [];
    
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

        // Collect documents instead of processing immediately
        if (results) {
          for (const doc of results) {
            // Skip if already processed in this run to avoid duplicates
            if (processedDocIds.has(doc.id)) {
              console.log(`Skipping duplicate document in this batch: ${doc.title} (${doc.id})`);
              documentsSkipped++;
              continue;
            }
            processedDocIds.add(doc.id);
            
            // Add to collection of documents
            allDocuments.push(doc);
          }
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
    
    // Process all documents in batches with TF-IDF context
    console.log(`\n=== PROCESSING ALL DOCUMENTS (${allDocuments.length}) ===\n`);
    await processDocuments(allDocuments, existingDocIds);
    
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