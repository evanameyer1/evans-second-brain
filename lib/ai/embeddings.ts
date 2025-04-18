// embeddings.ts  (OpenAI version)
import OpenAI from 'openai';
import { tokenLen } from './tokenizer';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // set this in your env
});

export async function safeEmbed(text: string, dim = 1536): Promise<number[]> {
  try {
    return await generateEmbeddings(text, dim);
  } catch (err: any) {
    // Only retry if we hit the 8192‑token error
    const tooLong = err?.status === 400;

    if (!tooLong) throw err; // different error → re‑throw

    console.log('safeEmbed error', err);
    console.log('text', text);

    // Split the string in half (by sentence if possible) & embed each half
    const mid = Math.floor(text.length / 2);
    const splitPoint =
      text.lastIndexOf('.', mid) > 100 ? text.lastIndexOf('.', mid) + 1 : mid;

    const left = text.slice(0, splitPoint).trim();
    const right = text.slice(splitPoint).trim();

    // recurse on each half
    const [vL, vR] = await Promise.all([
      safeEmbed(left, dim),
      safeEmbed(right, dim),
    ]);

    // Return the *average* of the two half‑vectors so lengths stay the same
    return vL.map((v, i) => (v + vR[i]) / 2);
  }
}

/* ---------- helper: embed MANY strings safely ---------- */
export async function safeEmbedBatch(texts: string[], dim = 1536) {
  const out: number[][] = [];
  for (const t of texts) out.push(await safeEmbed(t, dim));
  return out;
}

export async function generateEmbeddingsBatch(
  texts: string[],
  dim = 1536,
): Promise<number[][]> {
  const result: number[][] = [];
  let batch: string[] = [];
  let tokSum = 0;

  const flush = async () => {
    if (!batch.length) return;
    try {
      const { data } = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: batch,
        dimensions: dim,
        encoding_format: 'float',
      });
      result.push(...data.map((d) => d.embedding));
    } catch (err: any) {
      // fall back to per‑item safeEmbed on context‑length errors
      if (/maximum context length/i.test(err?.message ?? '')) {
        for (const s of batch) {
          result.push(await safeEmbed(s, dim));
        }
      } else {
        throw err;
      }
    }
    batch = [];
    tokSum = 0;
  };

  const limit = 8_192 - 32;
  for (const t of texts) {
    const toks = tokenLen(t);
    if (toks > limit) {
      // too big → let safeEmbed split it recursively
      result.push(await safeEmbed(t, dim));
      continue;
    }
    if (tokSum + toks > limit) await flush();
    batch.push(t);
    tokSum += toks;
  }
  await flush();
  return result;
}

/** ------------------------------------------------------------------
 *  Single‑text helper (kept for API compatibility)
 * ------------------------------------------------------------------ */
export async function generateEmbeddings(
  text: string,
  dim = 1536,
): Promise<number[]> {
  console.log(`\n=== Single Embedding ===`);
  console.log(`Text length: ${text.length} chars`);
  console.log(`Target dimension: ${dim}`);

  const { data } = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
    encoding_format: 'float',
  });

  const vec = data[0].embedding;
  console.log(`Generated embedding length: ${vec.length}`);
  console.log('========================\n');

  return vec.length < dim ? [...vec, ...Array(dim - vec.length).fill(0)] : vec;
}
