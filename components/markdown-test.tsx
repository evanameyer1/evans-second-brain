'use client';

import { useState } from 'react';
import { NonMemoizedMarkdown } from './markdown';

// Sample markdown with various elements that could cause nesting issues
const sampleMarkdown = `
# Markdown Rendering Test

This is a paragraph with **bold** and *italic* text.

## Code Block Test
Here's a code block:

\`\`\`javascript
const test = () => {
  console.log('Testing code blocks');
  return true;
};
\`\`\`

## List Test
- Item 1
- Item 2
  - Nested item with \`inline code\`
- Item 3 with a code block:
  \`\`\`
  const nestedCode = true;
  \`\`\`

## Blockquote Test
> This is a blockquote
> With multiple lines
> And some **formatting**

## Complex Nesting Test
- List item with a nested code block:
  \`\`\`
  // This code block is inside a list item
  function test() {
    return 'Should not be in a <p>';
  }
  \`\`\`
- Another item
`;

export default function MarkdownTest() {
  const [markdown, setMarkdown] = useState(sampleMarkdown);
  
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Markdown Rendering Test</h1>
      
      <div className="mb-4">
        <textarea 
          className="w-full h-64 p-2 border rounded" 
          value={markdown}
          onChange={(e) => setMarkdown(e.target.value)}
        />
      </div>
      
      <div className="border p-4 rounded">
        <h2 className="text-xl font-semibold mb-2">Rendered Output:</h2>
        <div className="markdown-content">
          <NonMemoizedMarkdown>{markdown}</NonMemoizedMarkdown>
        </div>
      </div>
    </div>
  );
} 