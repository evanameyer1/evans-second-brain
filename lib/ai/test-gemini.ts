/**
 * Simple test script for Gemini API
 */
import 'dotenv/config';
import { testGeminiConnection } from './readwise-search';

async function main() {
  console.log("Starting Gemini connection test...");
  
  try {
    const success = await testGeminiConnection();
    
    if (success) {
      console.log("✅ Gemini connection test successful!");
    } else {
      console.log("❌ Gemini connection test failed!");
      process.exit(1);
    }
  } catch (error) {
    console.error("Error running test:", error);
    process.exit(1);
  }
}

// Run test if called directly
if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch(err => {
      console.error("Unhandled error:", err);
      process.exit(1);
    });
}

export { main as testGemini }; 