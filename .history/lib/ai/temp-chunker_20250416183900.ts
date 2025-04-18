/**
 * Split text into chunks, merging paragraphs with overlap between chunks
 * @param text Full document text
 * @param maxTokens Maximum tokens per chunk (default: 512)
 * @param sentenceOverlap Number of sentences to overlap (default: 2)
 * @returns Array of chunks
 */
export function chunkTextWithOverlap(
    text: string,
    maxTokens = 2000,
    sentenceOverlap = 3
  ): string[] {
    // Split by paragraphs (one or more newlines)
    const paragraphs = text
      .split(/\n{2,}/)
      .map(p => p.trim())
      .filter(Boolean);
    
    // Simple token estimator (words × 1.3)
    const estimateTokens = (text: string): number => {
      return Math.ceil(text.split(/\s+/).length * 1.3);
    };
    
    // Split a paragraph into sentences
    const splitSentences = (paragraph: string): string[] => {
      // Match sentence endings (period, question mark, exclamation point)
      // followed by whitespace or end of string
      const matches = paragraph.match(/[^.!?]+[.!?]+\s*|[^.!?]+$/g);
      return matches || [paragraph];
    };
    
    const chunks: string[] = [];
    let currentChunk: string[] = [];
    let currentTokenCount = 0;
    let lastSentences: string[] = [];
    
    // Process each paragraph
    for (let i = 0; i < paragraphs.length; i++) {
      const paragraph = paragraphs[i];
      const paragraphTokens = estimateTokens(paragraph);
      
      // If paragraph alone is too big, split it by sentences
      if (paragraphTokens > maxTokens) {
        const sentences = splitSentences(paragraph);
        let tempChunk = "";
        
        for (const sentence of sentences) {
          const sentenceTokens = estimateTokens(sentence);
          
          if (estimateTokens(tempChunk + sentence) > maxTokens) {
            if (tempChunk) {
              chunks.push(tempChunk.trim());
              
              // Keep track of last sentences for overlap
              const tempSentences = splitSentences(tempChunk);
              lastSentences = tempSentences.slice(-sentenceOverlap);
            }
            
            tempChunk = sentence;
          } else {
            tempChunk += " " + sentence;
          }
        }
        
        if (tempChunk) {
          chunks.push(tempChunk.trim());
          const tempSentences = splitSentences(tempChunk);
          lastSentences = tempSentences.slice(-sentenceOverlap);
        }
        
        continue;
      }
      
      // Check if adding this paragraph would exceed the token limit
      if (currentTokenCount + paragraphTokens > maxTokens) {
        // Finalize current chunk with proper spacing
        chunks.push(currentChunk.join("\n\n").trim());
        
        // Extract last sentences from the current chunk for overlap
        const lastParagraph = currentChunk[currentChunk.length - 1];
        lastSentences = splitSentences(lastParagraph).slice(-sentenceOverlap);
        
        // Start a new chunk with the overlap
        currentChunk = lastSentences.length > 0 ? [lastSentences.join(" ")] : [];
        currentTokenCount = estimateTokens(lastSentences.join(" "));
      }
      
      // Add paragraph to current chunk
      currentChunk.push(paragraph);
      currentTokenCount += paragraphTokens;
    }
    
    // Add the last chunk if not empty
    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join("\n\n").trim());
    }
    
    return chunks;
  } 