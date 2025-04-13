#!/usr/bin/env ts-node
/**
 * Create a hybrid Pinecone index with support for dense and sparse vectors
 * Run this once to set up the index before using hybrid search
 * 
 * Usage: 
 * - Add PINECONE_API_KEY to .env file
 * - Run: npx ts-node scripts/create-hybrid-index.ts
 */

import { Pinecone } from '@pinecone-database/pinecone';
import 'dotenv/config';

async function createHybridIndex() {
  try {
    console.log('Creating hybrid Pinecone index...');

    const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
    
    // Check if index already exists
    const indexes = await pc.listIndexes();
    const indexNames = indexes.indexes?.map(idx => idx.name) || [];
    if (indexNames.includes('reader-embeddings-hybrid')) {
      console.log('Index "reader-embeddings-hybrid" already exists');
      return;
    }

    // Create the hybrid index
    await pc.createIndex({
      name: 'reader-embeddings-hybrid',
      dimension: 1024,             // size of llama‑text‑embed‑v2 vectors
      metric: 'dotproduct',        // Required for hybrid
      spec: { 
        serverless: { 
          cloud: 'aws', 
          region: 'us-east-1' 
        }
      }
    });

    console.log('Successfully created hybrid index "reader-embeddings-hybrid"');
    console.log('Update your .env file with:');
    console.log('PINECONE_INDEX=reader-embeddings-hybrid');
    
  } catch (error) {
    console.error('Error creating hybrid index:', error);
  }
}

createHybridIndex(); 