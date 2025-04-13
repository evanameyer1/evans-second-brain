#!/usr/bin/env ts-node
/**
 * Migrate existing data from a regular index to a hybrid index
 * Processes ALL documents in the old index, re-embeds them, and stores them with sparse vectors
 * 
 * Usage:
 * - First create the hybrid index with create-hybrid-index.ts
 * - Run: OLDINDEX=reader-embeddings NEWINDEX=reader-embeddings-hybrid npx ts-node scripts/migrate-to-hybrid.ts
 */

import { Pinecone } from '@pinecone-database/pinecone';
import 'dotenv/config';
import { toSparseVector } from '../lib/ai/sparse';
import { stripStops } from '../lib/ai/stopwords';

const EMBED_MODEL = 'llama-text-embed-v2';
const BATCH_SIZE = 100;

async function migrateToHybrid() {
  try {
    console.log('Starting migration to hybrid index...');
    
    // Source and destination indexes
    const oldIndexName = process.env.OLDINDEX || 'reader-embeddings';
    const newIndexName = process.env.NEWINDEX || 'reader-embeddings-hybrid';
    
    console.log(`Source: ${oldIndexName} → Destination: ${newIndexName}`);
    
    const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
    
    // Initialize old and new indexes
    const oldIndex = pc.index(oldIndexName);
    const newIndex = pc.index(newIndexName);
    
    // Check if both indexes exist
    try {
      await oldIndex.describeIndexStats();
      await newIndex.describeIndexStats();
    } catch (error) {
      console.error('Error: One of the indexes does not exist:', error);
      return;
    }
    
    console.log('Both indexes exist, continuing...');
    
    // Get total count
    const stats = await oldIndex.describeIndexStats();
    const totalRecords = stats.totalRecordCount || 0;
    console.log(`Found ${totalRecords} records to migrate`);
    
    if (totalRecords === 0) {
      console.log('No records to migrate');
      return;
    }
    
    // Process vectors in batches using vector ID prefix filtering
    const zeroVector = Array(1024).fill(0);
    let processed = 0;
    let batchNum = 0;
    const totalBatches = Math.ceil(totalRecords / BATCH_SIZE);
    
    while (processed < totalRecords) {
      try {
        console.log(`Processing batch ${batchNum + 1}/${totalBatches}`);
        
        // Query batch from old index
        const queryResponse = await oldIndex.query({
          vector: zeroVector,
          topK: BATCH_SIZE,
          includeMetadata: true,
          includeValues: true,
          filter: {
            id: { $gte: `${batchNum * BATCH_SIZE}` }
          }
        });
        
        const matches = queryResponse.matches || [];
        if (matches.length === 0) {
          console.log('No more records to process');
          break;
        }
        
        // Process each record - create sparse vectors and prepare for upsert
        const vectors = [];
        for (const match of matches) {
          if (!match.metadata?.text || !match.values) {
            console.log(`Skipping record ${match.id} - missing metadata or values`);
            continue;
          }
          
          // Create sparse vector from the text
          const text = String(match.metadata.text);
          const isHeader = match.metadata.header === true;
          
          // Create the appropriate text input for sparse vector
          const vectorText = isHeader
            ? `${match.metadata.title || ''} ${match.metadata.author || ''} ${stripStops(text)}`
            : text;
            
          const sparseVector = toSparseVector(vectorText);
          
          // Create the vector record with both dense and sparse values
          vectors.push({
            id: match.id,
            values: match.values,
            sparseValues: sparseVector,
            metadata: match.metadata,
          });
        }
        
        // Upsert to new index
        if (vectors.length > 0) {
          console.log(`Upserting ${vectors.length} vectors to ${newIndexName}`);
          await newIndex.upsert(vectors);
        }
        
        processed += matches.length;
        batchNum++;
        
        console.log(`Processed ${processed}/${totalRecords} records (${Math.round(processed/totalRecords*100)}%)`);
        
        // Break if we've processed all records or made too many attempts
        if (processed >= totalRecords || batchNum > totalBatches + 10) {
          console.log('Migration complete!');
          break;
        }
      } catch (error) {
        console.error('Error during migration:', error);
        break;
      }
    }
    
    console.log(`Total records processed: ${processed}`);
    
  } catch (error) {
    console.error('Error migrating to hybrid index:', error);
  }
}

migrateToHybrid(); 