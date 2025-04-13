import Link from 'next/link';
import React, { memo, Children, isValidElement } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
// Let's remove rehypeRaw for now as it may cause issues with nesting
// import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import { CodeBlock } from './code-block';

// Custom sanitization schema that extends the default
const sanitizeSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    // Allow specific data attributes if needed
    '*': [...(defaultSchema.attributes?.['*'] || []), 'className', 'data-*']
  }
};

// Enhanced helper function to check if a child is a block element
const isBlockElement = (child: React.ReactNode): boolean => {
  if (!isValidElement(child)) return false;
  
  const type = child.type;
  
  // Handle string types (HTML elements)
  if (typeof type === 'string') {
    return ['pre', 'div', 'ol', 'ul', 'table', 'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(type);
  }
  
  // Handle function types (React components)
  if (typeof type === 'function') {
    // Check component name or displayName for known block components
    // Use type assertion to access displayName and name properties safely
    return (
      ((type as any).displayName === 'CodeBlock' || (type as any).name === 'CodeBlock') ||
      // Add other known block component checks here
      false
    );
  }
  
  // For object types, check if it's a ForwardRef component
  if (typeof type === 'object' && type !== null) {
    // @ts-ignore - ForwardRef components might have a displayName or render function
    const name = type.displayName || (type.render && type.render.name);
    return name === 'CodeBlock' || name?.includes('Block') || name?.includes('pre');
  }
  
  return false;
};

const components: Partial<Components> = {
  // @ts-expect-error
  code: CodeBlock,
  // Always wrap pre in a div to avoid nesting issues
  pre: ({ children, ...props }) => <div className="block w-full"><pre {...props}>{children}</pre></div>,
  // More robust paragraph handling
  p: ({ node, children, ...props }) => {
    // Improved check for block elements
    const childrenArray = Children.toArray(children);
    const containsBlockElements = childrenArray.some(child => isBlockElement(child));
    
    // If any child is a block element, or if there's a code block, use div instead
    return containsBlockElements ? (
      <div {...props}>{children}</div>
    ) : (
      <p {...props}>{children}</p>
    );
  },
  // Ensure list types are properly styled and don't get nested in paragraphs
  ol: ({ node, children, ...props }) => {
    return (
      <ol className="list-decimal list-outside ml-4 my-4" {...props}>
        {children}
      </ol>
    );
  },
  li: ({ node, children, ...props }) => {
    // Check if this list item contains block elements
    const childArray = Children.toArray(children);
    const containsBlockElements = childArray.some(child => isBlockElement(child));
    
    // Add appropriate styling based on content
    return (
      <li className={`py-1 ${containsBlockElements ? 'mb-2' : ''}`} {...props}>
        {children}
      </li>
    );
  },
  ul: ({ node, children, ...props }) => {
    return (
      <ul className="list-disc list-outside ml-4 my-4" {...props}>
        {children}
      </ul>
    );
  },
  // Handle blockquotes properly
  blockquote: ({ node, children, ...props }) => {
    return (
      <blockquote 
        className="pl-4 border-l-4 border-gray-300 dark:border-gray-600 my-4 italic" 
        {...props}
      >
        {children}
      </blockquote>
    );
  },
  strong: ({ node, children, ...props }) => {
    return (
      <span className="font-semibold" {...props}>
        {children}
      </span>
    );
  },
  a: ({ node, children, ...props }) => {
    return (
      // @ts-expect-error
      <Link
        className="text-blue-500 hover:underline"
        target="_blank"
        rel="noreferrer"
        {...props}
      >
        {children}
      </Link>
    );
  },
  h1: ({ node, children, ...props }) => {
    return (
      <h1 className="text-3xl font-semibold mt-6 mb-2" {...props}>
        {children}
      </h1>
    );
  },
  h2: ({ node, children, ...props }) => {
    return (
      <h2 className="text-2xl font-semibold mt-6 mb-2" {...props}>
        {children}
      </h2>
    );
  },
  h3: ({ node, children, ...props }) => {
    return (
      <h3 className="text-xl font-semibold mt-6 mb-2" {...props}>
        {children}
      </h3>
    );
  },
  h4: ({ node, children, ...props }) => {
    return (
      <h4 className="text-lg font-semibold mt-6 mb-2" {...props}>
        {children}
      </h4>
    );
  },
  h5: ({ node, children, ...props }) => {
    return (
      <h5 className="text-base font-semibold mt-6 mb-2" {...props}>
        {children}
      </h5>
    );
  },
  h6: ({ node, children, ...props }) => {
    return (
      <h6 className="text-sm font-semibold mt-6 mb-2" {...props}>
        {children}
      </h6>
    );
  },
};

// Improved preprocessing function to clean markdown content
const preprocessMarkdown = (content: string): string => {
  if (!content) return '';
  
  // Ensure code blocks have proper line breaks
  const processedContent = content
    // Make sure code blocks are properly separated from paragraphs
    .replace(/```(.*)\n/g, '\n\n```$1\n')
    .replace(/\n```\s*/g, '\n```\n\n')
    // Add proper spacing around lists
    .replace(/(\n[*-] .*\n)(?=[^*-\s])/g, '$1\n')
    .replace(/(\n\d+\. .*\n)(?=[^\d\s])/g, '$1\n')
    // Add spacing around block elements to prevent nesting
    .replace(/(\n.+\n)(?=```)/g, '$1\n')
    .replace(/```.*\n[\s\S]*?```/g, (match) => `\n\n${match}\n\n`);
  
  return processedContent;
};

const remarkPlugins = [remarkGfm];
// Removing rehypeRaw for now as it can cause nesting issues
// Using proper type for rehypePlugins to fix linter error
const rehypePlugins = [[rehypeSanitize, sanitizeSchema]] as any;

export const NonMemoizedMarkdown = ({ children }: { children: string }) => {
  // Preprocess the markdown content
  const processedContent = preprocessMarkdown(children);
  
  return (
    <ReactMarkdown 
      remarkPlugins={remarkPlugins} 
      rehypePlugins={rehypePlugins}
      components={components}
    >
      {processedContent}
    </ReactMarkdown>
  );
};

export const Markdown = memo(
  NonMemoizedMarkdown,
  (prevProps, nextProps) => prevProps.children === nextProps.children,
);
