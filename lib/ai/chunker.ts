import { Pinecone } from '@pinecone-database/pinecone';
import { generateEmbeddings } from './embeddings';
import { tokenLen as rawTokenLen } from './tokenizer';

const CTX_LIMIT = 8_192; // hard OpenAI limit
const SINGLE_LIMIT = CTX_LIMIT - 1000; // safety buffer

/* ——————————————————— token‑length cache ——————————————————— */
const tokCache = new Map<string, number>();
const tokLen = (s: string) => {
  let n = tokCache.get(s);
  if (n === undefined) {
    n = rawTokenLen(s);
    tokCache.set(s, n);
  }
  return n;
};

/* ——————————————————— helpers ——————————————————— */
const splitSentences = (p: string): string[] =>
  p.match(/[^.!?]+[.!?]+["')\]]*|\S+$/g)?.map((s) => s.trim()) ?? [p];

/* ——————————————————— header chunker ——————————————————— */
export function headerChunkText(text: string, maxTokens = 512): string[] {
  const paragraphs = text
    .split(/\r?\n\s*\r?\n+/)
    .map((p) => p.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  let buffer = '';

  const pushBuf = () => {
    if (buffer) {
      chunks.push(buffer.trim());
      buffer = '';
    }
  };

  for (const para of paragraphs) {
    if (tokLen(para) > maxTokens) {
      // always sentence‑split long paragraphs
      pushBuf();
      let sentBuf = '';
      for (const sent of splitSentences(para)) {
        const next = sentBuf ? `${sentBuf} ${sent}` : sent;
        if (tokLen(next) > maxTokens) {
          if (sentBuf) chunks.push(sentBuf.trim());
          sentBuf = sent;
        } else {
          sentBuf = next;
        }
      }
      if (sentBuf) chunks.push(sentBuf.trim());
      continue;
    }

    const candidate = buffer ? `${buffer}\n\n${para}` : para;
    if (tokLen(candidate) > maxTokens) {
      pushBuf();
      buffer = para;
    } else {
      buffer = candidate;
    }
  }
  pushBuf();
  return chunks;
}

/* ——————————————————— Pinecone client (unused here, kept for context) ——————————————————— */
const pc = new Pinecone({
  apiKey:
    process.env.PINECONE_API_KEY ??
    (() => {
      throw new Error('PINECONE_API_KEY is required');
    })(),
});

/* ——————————————————— cosine similarity ——————————————————— */
function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return magA && magB ? dot / (magA * magB) : 0;
}

/* ——————————————————— semantic chunker ——————————————————— */
export async function semanticChunksWithOverlap(
  text: string,
  {
    minTokens = 300,
    maxTokens = 800,
    windowSize = 1,
    threshold = 0.75,
  }: {
    minTokens?: number;
    maxTokens?: number;
    windowSize?: number;
    threshold?: number;
  } = {},
): Promise<string[]> {
  /* 1. Initial paragraph merge (respecting min/max token targets) */
  const rawParas = text
    .split(/\r?\n\s*\r?\n+/)
    .map((p) => p.trim())
    .filter(Boolean);

  const paragraphs: string[] = [];
  const queue: string[] = [...rawParas];

  while (queue.length) {
    const p = queue.shift() || '';
    if (tokLen(p) > SINGLE_LIMIT) {
      // sentence‑split *instead of* heuristic midpoint
      for (const s of splitSentences(p)) queue.unshift(s);
      continue;
    }

    if (!paragraphs.length) {
      paragraphs.push(p);
      continue;
    }

    const candidate = `${paragraphs.at(-1)}\n\n${p}`;
    if (tokLen(candidate) <= maxTokens && tokLen(candidate) <= SINGLE_LIMIT) {
      paragraphs[paragraphs.length - 1] = candidate;
    } else {
      paragraphs.push(p);
    }
  }

  /* 2. Create sliding windows */
  type Win = { winIdx: number; curr: string; next: string };
  const windows: Win[] = [];

  for (let i = 0; i < paragraphs.length - windowSize; i++) {
    const curr = paragraphs
      .slice(Math.max(0, i - windowSize + 1), i + 1)
      .join(' ');
    const next = paragraphs.slice(i + 1, i + 1 + windowSize).join(' ');

    // Check if either window exceeds maxTokens
    const currTokens = tokLen(curr);
    const nextTokens = tokLen(next);

    if (currTokens > maxTokens || nextTokens > maxTokens) {
      console.log('\n=== Window Size Check ===');
      console.log(`Window ${i + 1}:`);
      console.log('Current window tokens:', currTokens);
      console.log('Next window tokens:', nextTokens);
      console.log('Max tokens:', maxTokens);
      console.log('Decision: SKIP (window too large)');
      console.log('========================\n');
      continue;
    }

    windows.push({ winIdx: i, curr, next });
  }

  if (windows.length === 0) {
    console.log('No valid windows found - all windows exceeded max token size');
    return paragraphs.map((p) => p.trim());
  }

  /* 3. Collect *unique* texts & embed in parallel */
  const allTexts = [...new Set(windows.flatMap((w) => [w.curr, w.next]))];
  console.log(`\n=== Embedding ${allTexts.length} unique window strings ===`);

  const vectorsArr = await Promise.all(
    allTexts.map((t) => generateEmbeddings(t, 1536)),
  );
  const vecMap = Object.fromEntries(allTexts.map((t, i) => [t, vectorsArr[i]]));

  const vecAt = (w: Win, which: 'curr' | 'next') => vecMap[w[which]];

  /* 4. Walk paragraphs, decide on chunk breaks */
  const chunks: string[] = [];
  let current: string[] = [];
  let currentTok = 0;
  let wPtr = 0;

  const flush = () => {
    if (current.length) {
      chunks.push(current.join('\n\n').trim());
      current = [];
      currentTok = 0;
    }
  };

  for (let i = 0; i < paragraphs.length; i++) {
    const para = paragraphs[i];
    const pTok = tokLen(para);
    const isLast = i === paragraphs.length - 1;

    // paragraph too large even after earlier split guard
    if (pTok > maxTokens) {
      flush();
      for (const sent of splitSentences(para)) {
        if (tokLen(sent) <= maxTokens) chunks.push(sent.trim());
      }
      continue;
    }

    current.push(para);
    currentTok += pTok;

    if (currentTok < minTokens && !isLast) continue;
    if (currentTok > maxTokens) {
      flush();
      continue;
    }

    if (currentTok >= minTokens && !isLast) {
      const nextParaTokens = tokLen(windows[wPtr].next);
      const combinedTokens = currentTok + nextParaTokens;

      if (combinedTokens > maxTokens) {
        console.log('\n=== Token Size Check ===');
        console.log('Current tokens:', currentTok);
        console.log('Next window tokens:', nextParaTokens);
        console.log('Combined tokens:', combinedTokens);
        console.log('Max tokens:', maxTokens);
        console.log('Decision: SPLIT (would exceed max tokens)');
        console.log('========================\n');
        flush();
        wPtr++;
        continue;
      }

      const currVec = vecAt(windows[wPtr], 'curr');
      const nextVec = vecAt(windows[wPtr], 'next');
      const sim = cosineSimilarity(currVec, nextVec);

      console.log('\n=== Semantic Comparison ===');
      console.log(`Window ${wPtr + 1}/${windows.length}:`);
      console.log('Current window tokens:', tokLen(windows[wPtr].curr));
      console.log('Next window tokens:', tokLen(windows[wPtr].next));
      console.log('\nCurrent window text:');
      console.log(
        windows[wPtr].curr.substring(0, 200) +
          (windows[wPtr].curr.length > 200 ? '...' : ''),
      );
      console.log('\nNext window text:');
      console.log(
        windows[wPtr].next.substring(0, 200) +
          (windows[wPtr].next.length > 200 ? '...' : ''),
      );
      console.log('\nCosine similarity:', sim.toFixed(3));
      console.log('Threshold:', threshold);
      console.log('Decision:', sim < threshold ? 'SPLIT' : 'CONTINUE');
      console.log('===========================\n');

      wPtr++;

      if (sim < threshold) flush();
    }
  }
  flush();
  return chunks;
}
