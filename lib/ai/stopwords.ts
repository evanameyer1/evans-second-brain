/* lib/ai/stopwords.ts
   Central place for stop‑word removal */
import { removeStopwords, eng } from 'stopword';

/** Return the string with all English stop‑words removed
 *  and collapsed whitespace.  Keeps original casing. */
export function stripStops(text: string): string {
  // tokenise on non‑word chars, preserve punctuation by re‑joining later
  const tokens = text.split(/\b/);              // ["What", " ", "is", " ", ...]
  const cleaned = removeStopwords(tokens, eng);
  return cleaned.join('').replace(/\s+/g, ' ').trim();
} 