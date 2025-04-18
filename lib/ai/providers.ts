import { customProvider } from 'ai';
import { google } from '@ai-sdk/google';

const gemini = (id: string) =>
  google(id, {
    safetySettings: [
      { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_CIVIC_INTEGRITY', threshold: 'BLOCK_NONE' },
    ],
  });

export const myProvider = customProvider({
  languageModels: {
    'chat-model': gemini('gemini-2.5-pro-exp-03-25'),
    'chat-model-reasoning': gemini('gemini-2.5-pro-exp-03-25'),
    'title-model': gemini('gemini-2.5-pro-exp-03-25'),
    'artifact-model': gemini('gemini-2.5-pro-exp-03-25'),
  },
});
