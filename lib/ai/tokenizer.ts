// tokenizer.ts
import { encode } from 'gpt-tokenizer';

/**
 * Return exact token count for a string using gpt-tokenizer
 */
export function tokenLen(text: string): number {
  return encode(text).length;
}

/** Split text so each part ≤ ctxLimit tokens (default 8192) */
export function splitToFit(text: string, ctx = 8192): string[] {
  if (tokenLen(text) <= ctx) return [text];
  const parts: string[] = [];
  let chunk = text;
  while (tokenLen(chunk) > ctx) {
    const mid = Math.floor(chunk.length / 2);
    const idx =
      chunk.lastIndexOf('.', mid) > 100 ? chunk.lastIndexOf('.', mid) + 1 : mid;
    parts.push(chunk.slice(0, idx).trim());
    chunk = chunk.slice(idx).trim();
  }
  parts.push(chunk);
  return parts;
}
