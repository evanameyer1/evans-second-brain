export const DEFAULT_CHAT_MODEL: string = 'chat-model';

interface ChatModel {
  id: string;
  name: string;
  description: string;
}

export const chatModels: Array<ChatModel> = [
  {
    id: 'readwise-only',
    name: 'Readwise notes',
    description: 'Answer only from your saved highlights',
  },
  {
    id: 'chat-model',
    name: 'Standard chatbot',
    description: 'General knowledge GPT',
  },
  {
    id: 'readwise-blend',
    name: 'Notes + GPT',
    description: 'Combine highlights with GPT context',
  }
];
