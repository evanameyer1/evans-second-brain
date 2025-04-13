"use strict";
/**
 * Keyword extraction utilities for document processing
 *
 * This file implements:
 * 1. RAKE (Rapid Automatic Keyword Extraction)
 * 2. A simple TF-IDF implementation
 * 3. Functions to extract and weight keywords for document headers
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SimpleTfIdf = void 0;
exports.addDocumentToTfIdf = addDocumentToTfIdf;
exports.buildTfIdfModel = buildTfIdfModel;
exports.extractTfIdfKeywords = extractTfIdfKeywords;
exports.extractRakeKeywords = extractRakeKeywords;
exports.boostTerm = boostTerm;
exports.extractKeywords = extractKeywords;
var stopword_1 = require("stopword");
// Simple TF-IDF implementation
var SimpleTfIdf = /** @class */ (function () {
    function SimpleTfIdf() {
        // Map of document IDs to term frequency maps
        this.documents = new Map();
        // Map of terms to document frequency
        this.documentFrequency = new Map();
        // Total number of documents
        this.documentCount = 0;
        // Built state flag
        this.built = false;
    }
    /**
     * Add a document to the TF-IDF model
     * @param docId Document ID
     * @param text Document text
     */
    SimpleTfIdf.prototype.addDocument = function (docId, text) {
        // Reset built state
        this.built = false;
        // Extract terms from text (simple tokenization)
        var terms = this.extractTerms(text);
        // Count term frequency in this document
        var termFreq = new Map();
        for (var _i = 0, terms_1 = terms; _i < terms_1.length; _i++) {
            var term = terms_1[_i];
            termFreq.set(term, (termFreq.get(term) || 0) + 1);
        }
        // Store document term frequencies
        this.documents.set(docId, termFreq);
        // Update document count
        this.documentCount++;
    };
    /**
     * Build the TF-IDF model by calculating document frequencies
     */
    SimpleTfIdf.prototype.build = function () {
        // Reset document frequency counts
        this.documentFrequency.clear();
        // Calculate document frequency for each term
        for (var _i = 0, _a = this.documents; _i < _a.length; _i++) {
            var _b = _a[_i], termFreqMap = _b[1];
            for (var _c = 0, _d = termFreqMap.keys(); _c < _d.length; _c++) {
                var term = _d[_c];
                this.documentFrequency.set(term, (this.documentFrequency.get(term) || 0) + 1);
            }
        }
        // Mark as built
        this.built = true;
    };
    /**
     * Get the TF-IDF score for a term in a document
     * @param docId Document ID
     * @param term Term to score
     * @returns TF-IDF score
     */
    SimpleTfIdf.prototype.tfIdf = function (docId, term) {
        // Ensure model is built
        if (!this.built) {
            throw new Error('TF-IDF model not built. Call build() first.');
        }
        // Check if document exists
        var termFreqMap = this.documents.get(docId);
        if (!termFreqMap) {
            return 0;
        }
        // Calculate TF (term frequency)
        var tf = termFreqMap.get(term) || 0;
        if (tf === 0) {
            return 0;
        }
        // Calculate IDF (inverse document frequency)
        var df = this.documentFrequency.get(term) || 0;
        if (df === 0) {
            return 0;
        }
        var idf = Math.log(this.documentCount / df);
        // Return TF-IDF score
        return tf * idf;
    };
    /**
     * Get the top N terms for a document by TF-IDF score
     * @param docId Document ID
     * @param n Number of terms to return
     * @returns Array of terms with scores
     */
    SimpleTfIdf.prototype.getTopTerms = function (docId, n) {
        if (n === void 0) { n = 10; }
        // Ensure model is built
        if (!this.built) {
            throw new Error('TF-IDF model not built. Call build() first.');
        }
        // Check if document exists
        var termFreqMap = this.documents.get(docId);
        if (!termFreqMap) {
            return [];
        }
        // Calculate TF-IDF for all terms in the document
        var termScores = [];
        for (var _i = 0, _a = termFreqMap.keys(); _i < _a.length; _i++) {
            var term = _a[_i];
            var score = this.tfIdf(docId, term);
            // Only consider meaningful terms (at least 3 chars and not just numbers)
            if (term.length >= 3 && !/^\d+$/.test(term)) {
                termScores.push({ term: term, weight: score });
            }
        }
        // Sort by score (descending) and return top N
        return termScores
            .sort(function (a, b) { return b.weight - a.weight; })
            .slice(0, n);
    };
    /**
     * Extract terms from text
     * @param text Input text
     * @returns Array of terms
     */
    SimpleTfIdf.prototype.extractTerms = function (text) {
        // Normalize text
        var normalizedText = text.toLowerCase()
            .replace(/[^\w\s]/g, ' ') // Replace non-alphanumeric with space
            .replace(/\s+/g, ' ') // Replace multiple spaces with single space
            .trim();
        // Split into words
        var words = normalizedText.split(' ');
        // Remove stopwords
        var termsWithoutStopwords = (0, stopword_1.removeStopwords)(words, stopword_1.eng);
        // Filter out very short terms
        return termsWithoutStopwords.filter(function (term) { return term.length >= 2; });
    };
    return SimpleTfIdf;
}());
exports.SimpleTfIdf = SimpleTfIdf;
// Global TF-IDF model
var globalTfIdf = new SimpleTfIdf();
/**
 * Add a document to the global TF-IDF model
 * @param docId Document ID
 * @param text Document text
 */
function addDocumentToTfIdf(docId, text) {
    globalTfIdf.addDocument(docId, text);
}
/**
 * Build the global TF-IDF model
 */
function buildTfIdfModel() {
    globalTfIdf.build();
}
/**
 * Get top TF-IDF terms for a document
 * @param docId Document ID
 * @param count Number of terms to return
 * @returns Array of top terms with weights
 */
function extractTfIdfKeywords(docId, count) {
    if (count === void 0) { count = 20; }
    return globalTfIdf.getTopTerms(docId, count);
}
/**
 * Extract keywords using RAKE algorithm
 * @param text Document text
 * @param topN Number of keywords to return
 * @returns Array of keywords with scores
 */
function extractRakeKeywords(text, topN) {
    if (topN === void 0) { topN = 20; }
    // Step 1: Normalize text
    var normalizedText = text.toLowerCase()
        .replace(/\n/g, '. ') // Replace newlines with periods
        .replace(/[^\w\s\.\,\-]/g, ' ') // Replace special chars with space
        .replace(/\s+/g, ' ') // Replace multiple spaces with single space
        .trim();
    // Step 2: Split text into phrases (candidate keywords)
    // Use periods, commas as phrase delimiters
    var phrases = normalizedText
        .split(/[\.\,]/)
        .map(function (phrase) { return phrase.trim(); })
        .filter(function (phrase) { return phrase.length > 0; });
    // Step 3: Extract keywords from phrases
    var candidateKeywords = new Map();
    var wordScores = calculateWordScores(phrases);
    // Extract keyword phrases and calculate scores
    for (var _i = 0, phrases_1 = phrases; _i < phrases_1.length; _i++) {
        var phrase = phrases_1[_i];
        // Skip if phrase is too short
        if (phrase.length < 3)
            continue;
        // Remove stopwords
        var words = phrase.split(' ');
        var filteredWords = (0, stopword_1.removeStopwords)(words, stopword_1.eng);
        // Skip empty phrases
        if (filteredWords.length === 0)
            continue;
        // Create candidate keyword
        var candidate = filteredWords.join(' ');
        // Skip if already processed
        if (candidateKeywords.has(candidate))
            continue;
        // Calculate score as sum of word scores
        var score = 0;
        for (var _a = 0, filteredWords_1 = filteredWords; _a < filteredWords_1.length; _a++) {
            var word = filteredWords_1[_a];
            score += wordScores.get(word) || 0;
        }
        // Add to candidates if meaningful
        if (score > 0 && candidate.length >= 3) {
            candidateKeywords.set(candidate, score);
        }
    }
    // Sort and convert to array
    var sortedKeywords = Array.from(candidateKeywords.entries())
        .map(function (_a) {
        var term = _a[0], weight = _a[1];
        return ({ term: term, weight: weight });
    })
        .sort(function (a, b) { return b.weight - a.weight; })
        .slice(0, topN);
    return sortedKeywords;
}
/**
 * Calculate word scores for RAKE algorithm
 * @param phrases Array of phrases
 * @returns Map of words to scores
 */
function calculateWordScores(phrases) {
    // Count word frequency and degree
    var wordFrequency = new Map();
    var wordDegree = new Map();
    for (var _i = 0, phrases_2 = phrases; _i < phrases_2.length; _i++) {
        var phrase = phrases_2[_i];
        var words = phrase.split(' ');
        var filteredWords = (0, stopword_1.removeStopwords)(words, stopword_1.eng);
        // Skip empty phrases
        if (filteredWords.length === 0)
            continue;
        // Update word frequency
        for (var _a = 0, filteredWords_2 = filteredWords; _a < filteredWords_2.length; _a++) {
            var word = filteredWords_2[_a];
            // Skip very short words
            if (word.length < 2)
                continue;
            wordFrequency.set(word, (wordFrequency.get(word) || 0) + 1);
            // Update word degree (co-occurrence with other words)
            wordDegree.set(word, (wordDegree.get(word) || 0) + filteredWords.length);
        }
    }
    // Calculate word scores
    var wordScores = new Map();
    for (var _b = 0, _c = wordFrequency.entries(); _b < _c.length; _b++) {
        var _d = _c[_b], word = _d[0], freq = _d[1];
        var degree = wordDegree.get(word) || 0;
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
function boostTerm(term, weight, maxRepeats) {
    if (maxRepeats === void 0) { maxRepeats = 5; }
    // Normalize weight to 1-maxRepeats range
    var repeats = Math.min(Math.max(Math.ceil(weight * 3), 1), maxRepeats);
    // Repeat the term based on weight
    return Array(repeats).fill(term).join(' ');
}
/**
 * Extract all keywords (RAKE + TF-IDF) for a document
 * @param docId Document ID
 * @param text Document text
 * @returns Object with keywords and boosted text
 */
function extractKeywords(docId, text) {
    // Extract RAKE keywords
    var rakeKeywords = extractRakeKeywords(text, 20);
    // Extract TF-IDF keywords
    var tfidfKeywords = extractTfIdfKeywords(docId, 20);
    // Create boosted text by repeating important terms
    // Boost more important terms by repeating them
    var boostedTerms = [];
    // Add RAKE terms (weighted)
    for (var _i = 0, rakeKeywords_1 = rakeKeywords; _i < rakeKeywords_1.length; _i++) {
        var _a = rakeKeywords_1[_i], term = _a.term, weight = _a.weight;
        // Normalize weight to 0-1 range
        var normalizedWeight = Math.min(weight / 10, 1);
        boostedTerms.push(boostTerm(term, normalizedWeight));
    }
    // Add TF-IDF terms (weighted)
    for (var _b = 0, tfidfKeywords_1 = tfidfKeywords; _b < tfidfKeywords_1.length; _b++) {
        var _c = tfidfKeywords_1[_b], term = _c.term, weight = _c.weight;
        // Normalize weight to 0-1 range
        var normalizedWeight = Math.min(weight * 2, 1);
        boostedTerms.push(boostTerm(term, normalizedWeight));
    }
    // Combine boosted terms
    var boostedText = boostedTerms.join(' ');
    return {
        rakeKeywords: rakeKeywords,
        tfidfKeywords: tfidfKeywords,
        boostedText: boostedText
    };
}
