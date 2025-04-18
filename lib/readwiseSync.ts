import { Pinecone } from '@pinecone-database/pinecone';
// Replace tiktoken with a simpler tokenization approach
import 'dotenv/config';
import { semanticChunksWithOverlap } from './ai/chunker';
import { toSparseVector } from './ai/sparse';
import {
  extractRakeKeywords,
  extractTfIdfKeywords,
  addDocumentToTfIdf,
  buildTfIdfModel,
  boostTerm,
  extractKeywords,
} from './ai/extract-keywords';
import { generateEmbeddings } from './ai/embeddings';
import { tokenLen } from './ai/tokenizer';

// Update to use the correct Reader API endpoint
const RW_API = 'https://readwise.io/api/v3/list/';
const pc = new Pinecone({
  apiKey:
    process.env.PINECONE_API_KEY ??
    (() => {
      throw new Error('PINECONE_API_KEY is required');
    })(),
});
// Update to use reader-embeddings index
const index = pc.index(process.env.PINECONE_INDEX || 'reader-embeddings');

// Embedding model
const EMBED_MODEL = 'llama-text-embed-v2';

// Function to fetch all existing document IDs from Pinecone
async function fetchExistingDocumentIds(): Promise<Set<string>> {
  const existingIds = new Set<string>();

  try {
    console.log('Fetching existing document IDs from Pinecone...');

    // Get index stats to understand the size of the database
    const stats = await index.describeIndexStats();
    const totalVectors = stats.totalRecordCount || 0;

    console.log(`Pinecone index contains ${totalVectors} total vectors`);

    if (totalVectors === 0) {
      console.log('Index is empty, no existing documents to check');
      return existingIds;
    }

    // Since Pinecone requires filters or vectors, we'll use a zero vector approach
    console.log('Using zero vector query to fetch document IDs');

    try {
      // Create a zero vector with the same dimension as your index (1536, not 4096)
      const zeroVector = Array(1536).fill(0);

      // Query with the zero vector to get random results across the index
      const response = await index.query({
        vector: zeroVector,
        topK: 10000,
        includeMetadata: true,
      });

      const matches = response.matches || [];
      console.log(`Retrieved ${matches.length} vectors with zero vector query`);

      // Extract document IDs
      for (const match of matches) {
        if (match.metadata?.doc_id) {
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
      console.error('Error fetching document IDs:', error);
      console.log('Will proceed without document deduplication');
    }

    return existingIds;
  } catch (error) {
    console.error('Error in fetchExistingDocumentIds:', error);
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
  category: string;
  html_content?: string | null;
  content?: string | null;
  summary?: string | null;
  created_at: string;
  tags?: string[];
};

/**
 * Clean and normalize HTML into plain text with clear paragraph boundaries.
 * - Converts block tags (p, div, headings, li) into blank-line separators.
 * - Converts <br> into single line breaks.
 * - Strips all other tags.
 * - Decodes common HTML entities.
 * - Normalizes all newlines to '\n'.
 * - Collapses multiple blank lines into exactly two '\n\n' for paragraph splits.
 *
 * @param html  Raw HTML string
 * @returns     Cleaned text where each paragraph is separated by '\n\n'
 */
export function extractTextFromHtml(html: string): string {
  let text = html;

  // 1) Preserve paragraph boundaries
  text = text
    .replace(/<br\s*\/?>/gi, '\n') // <br> → newline
    .replace(/<(?:p|div|h[1-6]|li)[^>]*>/gi, '\n\n') // open block tags → blank lines
    .replace(/<\/(?:p|div|h[1-6]|li)>/gi, ''); // close block tags → removed

  // 2) Strip any remaining tags
  text = text.replace(/<[^>]+>/g, '');

  // 3) Decode common HTML entities
  text = text
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/g, "'");

  // 4) Normalize whitespace and newlines
  text = text
    .replace(/\r\n?/g, '\n') // unify Windows CRLF and CR to LF
    .replace(/[ \t]+(?=\n)/g, '') // trim trailing spaces on lines
    .replace(/[ \t]{2,}/g, ' ') // collapse multiple spaces
    .replace(/\n{3,}/g, '\n\n'); // collapse 3+ newlines into 2

  return text.trim();
}

async function processDocuments(
  docs: ReaderDoc[],
  existingIds: Set<string>,
): Promise<void> {
  // Phase 1: Build TF-IDF corpus
  for (const doc of docs) {
    if (existingIds.has(doc.id)) continue;
    const text = doc.html_content
      ? extractTextFromHtml(doc.html_content)
      : doc.content || '';
    if (text) addDocumentToTfIdf(doc.id, text);
  }
  buildTfIdfModel();

  // Phase 2: Process each document
  for (const doc of docs) {
    await processDocument(doc, existingIds);
  }
}

async function processDocument(
  doc: ReaderDoc,
  existingIds: Set<string>,
): Promise<void> {
  if (existingIds.has(doc.id)) return;

  console.log(`\n=== Processing Document ===`);
  console.log(`Title: ${doc.title}`);
  console.log(`ID: ${doc.id}`);

  /* ---------- 1. Extract and pre‑process text ---------- */
  const bodyText = doc.html_content
    ? extractTextFromHtml(doc.html_content)
    : doc.content || '';
  if (!bodyText) {
    console.log('No content found, skipping document');
    return;
  }

  console.log(`Body text length: ${bodyText.length} chars`);

  /* ---------- 2. Build "super‑header" text ---------- */
  const summary = doc.summary || '';
  const tags = Array.isArray(doc.tags)
    ? doc.tags
        .map((t) => (typeof t === 'string' ? t : String((t as any).name || t)))
        .join(', ')
    : '';
  const { rakeKeywords, tfidfKeywords } = extractKeywords(doc.id, bodyText);

  let superHeaderText = [
    `Title: ${doc.title.substring(0, 100)}${doc.title.length > 100 ? '...' : ''}`,
    `Author: ${doc.author.substring(0, 100)}${doc.author.length > 100 ? '...' : ''}`,
    tags && `Tags: ${tags.substring(0, 100)}${tags.length > 100 ? '...' : ''}`,
    summary &&
      `Summary: ${summary.substring(0, 1000)}${summary.length > 1000 ? '...' : ''}`,
    `RAKE Keywords: ${rakeKeywords.map((k) => k.term).join(', ')}`,
    `TF-IDF Terms: ${tfidfKeywords.map((k) => k.term).join(', ')}`,
  ]
    .filter(Boolean)
    .join('\n\n');

  superHeaderText = superHeaderText.substring(0, 1800);

  console.log(`Header text length: ${superHeaderText.length} chars`);

  /* ---------- 3. Chunk the body ---------- */
  console.log('\nStarting semantic chunking...');
  const chunks = await semanticChunksWithOverlap(bodyText, {
    minTokens: 300,
    maxTokens: 800,
    windowSize: 1,
    threshold: 0.75,
  });

  console.log(`Created ${chunks.length} chunks`);

  /* ---------- 4. Embed header + chunks in true batches ---------- */
  const texts = [superHeaderText, ...chunks];
  const vectors: number[][] = [];

  console.log('\nEmbedding header + chunks individually…');
  for (const [i, t] of texts.entries()) {
    console.log(`Embedding ${i + 1}/${texts.length}`);
    // Ensure we're not passing text that's too large to embed
    if (tokenLen(t) > 8000) {
      // OpenAI's max context length
      console.log(
        `Warning: Text ${i + 1} exceeds max token size, truncating...`,
      );
      const truncated = t.substring(0, 6000); // Rough estimate to stay under token limit
      vectors.push(await generateEmbeddings(truncated, 1536));
    } else {
      vectors.push(await generateEmbeddings(t, 1536));
    }
  }

  const headerVector = vectors[0];
  const chunkVectors = vectors.slice(1);

  console.log(`Generated ${vectors.length} vectors total`);

  /* ---------- 5. Sparse vector for header ---------- */
  const sparseHeader = toSparseVector(superHeaderText);

  /* ---------- 6. Upsert header ---------- */
  console.log('\nUpserting header vector...');
  await index.upsert([
    {
      id: `${doc.id}-header`,
      values: headerVector,
      sparseValues: sparseHeader,
      metadata: {
        doc_id: doc.id,
        title: doc.title,
        author: doc.author,
        url: doc.url,
        category: doc.category,
        summary,
        tags: tags.split(', ').filter(Boolean),
        created_at: doc.created_at,
        header: true,
      },
    },
  ]);

  /* ---------- 7. Upsert each chunk ---------- */
  console.log('\nUpserting chunk vectors...');
  for (let i = 0; i < chunks.length; i++) {
    console.log(`Upserting chunk ${i + 1}/${chunks.length}`);
    await index.upsert([
      {
        id: `${doc.id}-chunk-${i}`,
        values: chunkVectors[i],
        sparseValues: toSparseVector(chunks[i]),
        metadata: {
          doc_id: doc.id,
          title: doc.title,
          author: doc.author,
          url: doc.url,
          category: doc.category,
          text: chunks[i],
          header: false,
          chunk_id: i,
          created_at: doc.created_at,
        },
      },
    ]);
  }

  console.log('\nDocument processing complete');
  console.log('========================\n');
}

export async function syncReadwise(
  updatedAfter?: string,
  forceUpdate = false,
): Promise<void> {
  // Track processed documents to avoid duplicates across all locations
  const processedDocIds = new Set<string>();
  // Define locations to fetch
  const locations = ['new', 'later', 'archive'];

  try {
    console.log('Starting Readwise Reader sync process...');
    console.log(`API Token exists: ${Boolean(process.env.READWISE_TOKEN)}`);

    // Only fetch existing document IDs if we're not forcing an update
    let existingDocIds = new Set<string>();
    if (!forceUpdate) {
      existingDocIds = await fetchExistingDocumentIds();
      console.log(
        `Will skip ${existingDocIds.size} documents that already exist in Pinecone`,
      );
    } else {
      console.log(
        'Force update mode: Will process all documents regardless of existing vectors',
      );
    }

    // Track stats across all locations
    let totalDocumentsReceived = 0;
    let documentsSkipped = 0;
    const documentsProcessed = 0;
    const chunksEmbedded = 0;

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

        console.log(
          `Fetching from Readwise Reader API (${location}): ${url.replace(/Token [^&]+/, 'Token [REDACTED]')}`,
        );

        const res = await fetch(url, {
          headers: {
            Authorization: `Token ${process.env.READWISE_TOKEN}`,
            Accept: 'application/json',
          },
        });

        console.log(`Response status: ${res.status} ${res.statusText}`);

        if (res.status === 429) {
          const wait = +(res.headers.get('Retry-After') || '60') * 1000;
          console.log(`Rate limited, waiting ${wait}ms before retrying`);
          await new Promise((r) => setTimeout(r, wait));
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

        let data: { results: ReaderDoc[]; nextPageCursor?: string };
        try {
          data = (await res.json()) as {
            results: ReaderDoc[];
            nextPageCursor?: string;
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
        console.log(
          `Received ${batchSize} documents from Readwise Reader for ${location} (total across locations: ${totalDocumentsReceived})`,
        );

        // Collect documents instead of processing immediately
        if (results) {
          for (const doc of results) {
            // Skip if already processed in this run to avoid duplicates
            if (processedDocIds.has(doc.id)) {
              console.log(
                `Skipping duplicate document in this batch: ${doc.title} (${doc.id})`,
              );
              documentsSkipped++;
              continue;
            }
            processedDocIds.add(doc.id);

            // Add to collection of documents
            allDocuments.push(doc);
          }
        }

        if (!nextPageCursor) {
          console.log(
            `No more pages for location "${location}", moving to next location`,
          );
          break;
        }
        cursor = nextPageCursor;
        console.log(
          `Moving to next page for location "${location}" with cursor: ${cursor}`,
        );
      }

      // Location completion status
      console.log(`\n=== Completed location: ${location} ===\n`);
    }

    // Process all documents in batches with TF-IDF context
    console.log(
      `\n=== PROCESSING ALL DOCUMENTS (${allDocuments.length}) ===\n`,
    );
    await processDocuments(allDocuments, existingDocIds);

    // Log final stats
    console.log('\n=== SYNC SUMMARY ===');
    console.log(`Total documents received: ${totalDocumentsReceived}`);
    console.log(`Documents processed: ${documentsProcessed}`);
    console.log(`Documents skipped: ${documentsSkipped}`);
    console.log(`Chunks embedded: ${chunksEmbedded}`);
    console.log(`Locations processed: ${locations.join(', ')}`);
    console.log('====================');
  } catch (error) {
    console.error('Error in syncReadwise:', error);
    throw error;
  }
}
