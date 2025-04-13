# Setting Up Hybrid Retrieval for Readwise

This guide explains how to set up hybrid retrieval (dense + sparse vectors) for more accurate search results in your personal knowledge base.

## Why Hybrid Search?

- **Dense vectors** (embeddings) are good at finding semantic similarity (paraphrases, related concepts)
- **Sparse vectors** are good at exact term matching (technical terms, code snippets, IDs)
- **Hybrid search** combines both approaches for better results

## Setup Steps

1. **Create a new hybrid-compatible Pinecone index**

```bash
# Run the index creation script
npx ts-node scripts/create-hybrid-index.ts
```

This creates a new index called `reader-embeddings-hybrid` with:
- 1024 dimensions (for llama-text-embed-v2)
- dotproduct metric (required for hybrid search)
- serverless configuration

2. **Update your environment variables**

```bash
# In .env file
PINECONE_INDEX=reader-embeddings-hybrid
```

3. **Migrate existing data to the hybrid index**

```bash
# Run the migration script
npx ts-node scripts/migrate-to-hybrid.ts
```

This will:
- Read all documents from your existing index
- Create appropriate sparse vectors for each document
- Upsert both dense and sparse vectors to the new hybrid index

## Testing the Setup

Try a query that should benefit from exact term matching (like technical terms or code snippets) and see if the results improve.

## Error Troubleshooting

If you see this error:
```
Error: Index configuration does not support sparse values - only indexes that are sparse or using dotproduct are supported
```

It means your index isn't configured for hybrid search. Make sure you:
1. Created a new index with `metric: 'dotproduct'`
2. Are using the correct index name in your environment
3. Have waited for index initialization to complete (can take a few minutes)

## Code Changes

No changes needed to the query code - the current implementation already uses the correct parameter names:

```typescript
await index.query({
  vector: denseQueryValues,      // Dense vector (numbers)
  sparseVector: sparseQuery,     // Sparse vector (indices & values)
  // ... other parameters
});
``` 