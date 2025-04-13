/**
 * Keyword extraction utilities for document processing
 * 
 * This file implements:
 * 1. RAKE (Rapid Automatic Keyword Extraction)
 * 2. A simple TF-IDF implementation
 * 3. Functions to extract and weight keywords for document headers
 */

import { removeStopwords, eng as stopwords } from 'stopword';

// Simple TF-IDF implementation
export class SimpleTfIdf {
  // Map of document IDs to term frequency maps
  private documents: Map<string, Map<string, number>> = new Map();
  // Map of terms to document frequency
  private documentFrequency: Map<string, number> = new Map();
  // Total number of documents
  private documentCount: number = 0;
  // Built state flag
  private built: boolean = false;

  /**
   * Add a document to the TF-IDF model
   * @param docId Document ID
   * @param text Document text
   */
  addDocument(docId: string, text: string): void {
    // Reset built state
    this.built = false;
    
    // Extract terms from text (simple tokenization)
    const terms = this.extractTerms(text);
    
    // Count term frequency in this document
    const termFreq = new Map<string, number>();
    for (const term of terms) {
      termFreq.set(term, (termFreq.get(term) || 0) + 1);
    }
    
    // Store document term frequencies
    this.documents.set(docId, termFreq);
    
    // Update document count
    this.documentCount++;
  }

  /**
   * Build the TF-IDF model by calculating document frequencies
   */
  build(): void {
    // Reset document frequency counts
    this.documentFrequency.clear();
    
    // Calculate document frequency for each term
    for (const [, termFreqMap] of this.documents) {
      for (const term of termFreqMap.keys()) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) || 0) + 1);
      }
    }
    
    // Mark as built
    this.built = true;
  }

  /**
   * Get the TF-IDF score for a term in a document
   * @param docId Document ID
   * @param term Term to score
   * @returns TF-IDF score
   */
  tfIdf(docId: string, term: string): number {
    // Ensure model is built
    if (!this.built) {
      throw new Error('TF-IDF model not built. Call build() first.');
    }
    
    // Check if document exists
    const termFreqMap = this.documents.get(docId);
    if (!termFreqMap) {
      return 0;
    }
    
    // Calculate TF (term frequency)
    const tf = termFreqMap.get(term) || 0;
    if (tf === 0) {
      return 0;
    }
    
    // Calculate IDF (inverse document frequency)
    const df = this.documentFrequency.get(term) || 0;
    if (df === 0) {
      return 0;
    }
    
    const idf = Math.log(this.documentCount / df);
    
    // Return TF-IDF score
    return tf * idf;
  }

  /**
   * Get the top N terms for a document by TF-IDF score
   * @param docId Document ID
   * @param n Number of terms to return
   * @returns Array of terms with scores
   */
  getTopTerms(docId: string, n: number = 10): Array<{term: string, weight: number}> {
    // Ensure model is built
    if (!this.built) {
      throw new Error('TF-IDF model not built. Call build() first.');
    }
    
    // Check if document exists
    const termFreqMap = this.documents.get(docId);
    if (!termFreqMap) {
      return [];
    }
    
    // Calculate TF-IDF for all terms in the document
    const termScores: Array<{term: string, weight: number}> = [];
    for (const term of termFreqMap.keys()) {
      const score = this.tfIdf(docId, term);
      // Only consider meaningful terms (at least 3 chars and not just numbers)
      if (term.length >= 3 && !/^\d+$/.test(term)) {
        termScores.push({ term, weight: score });
      }
    }
    
    // Sort by score (descending) and return top N
    return termScores
      .sort((a, b) => b.weight - a.weight)
      .slice(0, n);
  }

  /**
   * Extract terms from text
   * @param text Input text
   * @returns Array of terms
   */
  private extractTerms(text: string): string[] {
    // Normalize text
    const normalizedText = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Replace non-alphanumeric with space
      .replace(/\s+/g, ' ')     // Replace multiple spaces with single space
      .trim();
    
    // Split into words
    const words = normalizedText.split(' ');
    
    // Remove stopwords
    const termsWithoutStopwords = removeStopwords(words, stopwords);
    
    // Filter out very short terms
    return termsWithoutStopwords.filter(term => term.length >= 2);
  }
}

// Global TF-IDF model
const globalTfIdf = new SimpleTfIdf();

/**
 * Add a document to the global TF-IDF model
 * @param docId Document ID
 * @param text Document text
 */
export function addDocumentToTfIdf(docId: string, text: string): void {
  globalTfIdf.addDocument(docId, text);
}

/**
 * Build the global TF-IDF model
 */
export function buildTfIdfModel(): void {
  globalTfIdf.build();
}

/**
 * Get top TF-IDF terms for a document
 * @param docId Document ID
 * @param count Number of terms to return
 * @returns Array of top terms with weights
 */
export function extractTfIdfKeywords(docId: string, count: number = 20): Array<{term: string, weight: number}> {
  return globalTfIdf.getTopTerms(docId, count);
}

/**
 * Extract keywords using RAKE algorithm
 * @param text Document text
 * @param topN Number of keywords to return
 * @returns Array of keywords with scores
 */
export function extractRakeKeywords(text: string, topN: number = 20): Array<{term: string, weight: number}> {
  // Step 1: Normalize text
  const normalizedText = text.toLowerCase()
    .replace(/\n/g, '. ')            // Replace newlines with periods
    .replace(/[^\w\s\.\,\-]/g, ' ')  // Replace special chars with space
    .replace(/\s+/g, ' ')            // Replace multiple spaces with single space
    .trim();
  
  // Step 2: Split text into phrases (candidate keywords)
  // Use periods, commas as phrase delimiters
  const phrases = normalizedText
    .split(/[\.\,]/)
    .map(phrase => phrase.trim())
    .filter(phrase => phrase.length > 0);
  
  // Step 3: Extract keywords from phrases
  const candidateKeywords = new Map<string, number>();
  const wordScores = calculateWordScores(phrases);
  
  // Extract keyword phrases and calculate scores
  for (const phrase of phrases) {
    // Skip if phrase is too short
    if (phrase.length < 3) continue;
    
    // Remove stopwords
    const words = phrase.split(' ');
    const filteredWords = removeStopwords(words, stopwords);
    
    // Skip empty phrases
    if (filteredWords.length === 0) continue;
    
    // Create candidate keyword
    const candidate = filteredWords.join(' ');
    
    // Skip if already processed
    if (candidateKeywords.has(candidate)) continue;
    
    // Calculate score as sum of word scores
    let score = 0;
    for (const word of filteredWords) {
      score += wordScores.get(word) || 0;
    }
    
    // Add to candidates if meaningful
    if (score > 0 && candidate.length >= 3) {
      candidateKeywords.set(candidate, score);
    }
  }
  
  // Sort and convert to array
  const sortedKeywords = Array.from(candidateKeywords.entries())
    .map(([term, weight]) => ({ term, weight }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, topN);
  
  return sortedKeywords;
}

/**
 * Calculate word scores for RAKE algorithm
 * @param phrases Array of phrases
 * @returns Map of words to scores
 */
function calculateWordScores(phrases: string[]): Map<string, number> {
  // Count word frequency and degree
  const wordFrequency = new Map<string, number>();
  const wordDegree = new Map<string, number>();
  
  for (const phrase of phrases) {
    const words = phrase.split(' ');
    const filteredWords = removeStopwords(words, stopwords);
    
    // Skip empty phrases
    if (filteredWords.length === 0) continue;
    
    // Update word frequency
    for (const word of filteredWords) {
      // Skip very short words
      if (word.length < 2) continue;
      
      wordFrequency.set(word, (wordFrequency.get(word) || 0) + 1);
      
      // Update word degree (co-occurrence with other words)
      wordDegree.set(word, (wordDegree.get(word) || 0) + filteredWords.length);
    }
  }
  
  // Calculate word scores
  const wordScores = new Map<string, number>();
  for (const [word, freq] of wordFrequency.entries()) {
    const degree = wordDegree.get(word) || 0;
    wordScores.set(word, degree / freq);
  }
  
  return wordScores;
}

/**
 * Boost a term by repeating it based on weight
 * @param term Term to boost
 * @param weight Weight/importance of the term
 * @param maxRepeats Maximum number of times to repeat
 * @returns Boosted term string
 */
export function boostTerm(term: string, weight: number, maxRepeats: number = 5): string {
  // Normalize weight to 1-maxRepeats range
  const repeats = Math.min(Math.max(Math.ceil(weight * 3), 1), maxRepeats);
  
  // Repeat the term based on weight
  return Array(repeats).fill(term).join(' ');
}

/**
 * Extract all keywords (RAKE + TF-IDF) for a document
 * @param docId Document ID
 * @param text Document text
 * @returns Object with keywords and boosted text
 */
export function extractKeywords(docId: string, text: string): {
  rakeKeywords: Array<{term: string, weight: number}>,
  tfidfKeywords: Array<{term: string, weight: number}>,
  boostedText: string
} {
  // Extract RAKE keywords
  const rakeKeywords = extractRakeKeywords(text, 20);
  
  // Extract TF-IDF keywords
  const tfidfKeywords = extractTfIdfKeywords(docId, 20);
  
  // Create boosted text by repeating important terms
  // Boost more important terms by repeating them
  const boostedTerms: string[] = [];
  
  // Add RAKE terms (weighted)
  for (const { term, weight } of rakeKeywords) {
    // Normalize weight to 0-1 range
    const normalizedWeight = Math.min(weight / 10, 1);
    boostedTerms.push(boostTerm(term, normalizedWeight));
  }
  
  // Add TF-IDF terms (weighted)
  for (const { term, weight } of tfidfKeywords) {
    // Normalize weight to 0-1 range
    const normalizedWeight = Math.min(weight * 2, 1);
    boostedTerms.push(boostTerm(term, normalizedWeight));
  }
  
  // Combine boosted terms
  const boostedText = boostedTerms.join(' ');
  
  return {
    rakeKeywords,
    tfidfKeywords,
    boostedText
  };
} 