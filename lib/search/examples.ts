/**
 * Example usage of the two-stage search implementation
 */

import 'dotenv/config';
import { twoStageSearch, enhancedTwoStageSearch } from './twoStageSearch';

/**
 * Simple example of using the original two-stage search
 */
async function searchExample() {
  try {
    // Example query
    const query = "SQL database optimization techniques";
    
    // Execute two-stage search
    console.log(`Searching for: "${query}"`);
    const results = await twoStageSearch(
      query,
      10,  // Stage 1: Consider top 10 documents
      5    // Stage 2: Return top 5 chunks
    );
    
    // Print results
    console.log(`Found ${results.length} relevant chunks`);
    results.forEach((result, i) => {
      console.log(`\nResult ${i + 1} (score: ${result.score.toFixed(3)})`);
      console.log(`Title: ${result.title}`);
      console.log(`URL: ${result.url}`);
      console.log(`Text snippet: ${result.text.substring(0, 150)}...`);
    });
    
  } catch (error) {
    console.error('Search error:', error);
  }
}

/**
 * Example of using the Gemini-enhanced two-stage search
 */
async function enhancedSearchExample() {
  try {
    // Example query
    const query = "SQL database optimization techniques";
    
    // Execute enhanced two-stage search with Gemini query rewriting
    console.log(`Enhanced searching for: "${query}"`);
    const results = await enhancedTwoStageSearch(
      query,
      5    // Return top 5 chunks
    );
    
    // Print results
    console.log(`Found ${results.length} relevant chunks with enhanced search`);
    results.forEach((result, i) => {
      console.log(`\nResult ${i + 1} (score: ${result.score.toFixed(3)})`);
      console.log(`Title: ${result.title}`);
      console.log(`Text snippet: ${result.text.substring(0, 150)}...`);
    });
    
  } catch (error) {
    console.error('Enhanced search error:', error);
  }
}

/**
 * Compare both search methods side by side
 */
async function compareSearchMethods() {
  try {
    const query = "SQL database optimization techniques";
    
    console.log("=== COMPARING SEARCH METHODS ===");
    console.log(`Query: "${query}"`);
    console.log("===============================");
    
    // Run both search methods
    const [standardResults, enhancedResults] = await Promise.all([
      twoStageSearch(query, 10, 5),
      enhancedTwoStageSearch(query, 5)
    ]);
    
    // Print standard results
    console.log("\n=== STANDARD SEARCH RESULTS ===");
    console.log(`Found ${standardResults.length} results`);
    standardResults.forEach((result, i) => {
      console.log(`\n${i + 1}. ${result.title} (${result.score.toFixed(3)})`);
      console.log(`   ${result.text.substring(0, 100)}...`);
    });
    
    // Print enhanced results
    console.log("\n=== ENHANCED SEARCH RESULTS ===");
    console.log(`Found ${enhancedResults.length} results`);
    enhancedResults.forEach((result, i) => {
      console.log(`\n${i + 1}. ${result.title} (${result.score.toFixed(3)})`);
      console.log(`   ${result.text.substring(0, 100)}...`);
    });
    
    console.log("\n===============================");
  } catch (error) {
    console.error('Search comparison error:', error);
  }
}

/**
 * Example of using the two-stage search as part of a prompt augmentation workflow
 */
async function promptAugmentationExample(query: string): Promise<string> {
  try {
    // Execute enhanced two-stage search for better results
    console.log(`Finding context for query: "${query}"`);
    const results = await enhancedTwoStageSearch(query, 3);
    
    if (results.length === 0) {
      return "No relevant information found.";
    }
    
    // Format results as context for an LLM prompt
    const context = results.map((result, i) => {
      return `[Document ${i + 1}: ${result.title}]\n${result.text}\n`;
    }).join('\n');
    
    // Example prompt template (you would send this to your LLM)
    return `
Here is some relevant information about "${query}":

${context}

Based on the information above, please answer the following question:
${query}
`;
  } catch (error) {
    console.error('Prompt augmentation error:', error);
    return "An error occurred while retrieving context information.";
  }
}

// Execute the example if this file is run directly
if (require.main === module) {
  compareSearchMethods()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error('Example error:', error);
      process.exit(1);
    });
}

export { searchExample, enhancedSearchExample, compareSearchMethods, promptAugmentationExample }; 