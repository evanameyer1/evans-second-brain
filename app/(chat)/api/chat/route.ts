import type { UIMessage } from 'ai';
import {
  appendResponseMessages,
  createDataStreamResponse,
  smoothStream,
  streamText,
} from 'ai';
import { auth } from '@/app/(auth)/auth';
import {
  systemPrompt,
  readwiseOnlyPrompt,
  readwiseBlendPrompt,
} from '@/lib/ai/prompts';
import {
  deleteChatById,
  getChatById,
  saveChat,
  saveMessages,
  getMessagesByChatId,
} from '@/lib/db/queries';
import {
  generateUUID,
  getMostRecentUserMessage,
  getTrailingMessageId,
} from '@/lib/utils';
import { generateTitleFromUserMessage } from '../../actions';
import { createDocument } from '@/lib/ai/tools/create-document';
import { updateDocument } from '@/lib/ai/tools/update-document';
import { requestSuggestions } from '@/lib/ai/tools/request-suggestions';
import { getWeather } from '@/lib/ai/tools/get-weather';
import { isProductionEnvironment } from '@/lib/constants';
import { myProvider } from '@/lib/ai/providers';
import { hybridSearch, formatReadwiseContext } from '@/lib/ai/readwise-search';

export const maxDuration = 60;

// Helper function for timestamped logging
function logWithTimestamp(message: string, data?: any) {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
  if (data) {
    console.log(`[${timestamp}] Data:`, data);
  }
}

export async function POST(request: Request) {
  logWithTimestamp('Starting chat POST request');
  try {
    const {
      id,
      messages,
      selectedChatModel,
    }: {
      id: string;
      messages: Array<UIMessage>;
      selectedChatModel: string;
    } = await request.json();

    logWithTimestamp('Request parsed', {
      id,
      selectedChatModel,
      messageCount: messages.length,
    });

    const session = await auth();

    if (!session || !session.user || !session.user.id) {
      logWithTimestamp('Auth failed: Unauthorized session');
      return new Response('Unauthorized', { status: 401 });
    }

    logWithTimestamp('User authenticated', { userId: session.user.id });
    const userMessage = getMostRecentUserMessage(messages);

    if (!userMessage) {
      logWithTimestamp('Error: No user message found in request');
      return new Response('No user message found', { status: 400 });
    }

    logWithTimestamp('Latest user message', {
      messageId: userMessage.id,
      preview: JSON.stringify(userMessage).substring(0, 200),
    });

    const chat = await getChatById({ id });

    if (!chat) {
      logWithTimestamp('Creating new chat');
      const title = await generateTitleFromUserMessage({
        message: userMessage,
      });
      logWithTimestamp('Generated chat title', { title });

      await saveChat({ id, userId: session.user.id, title });
      logWithTimestamp('New chat saved');
    } else {
      logWithTimestamp('Using existing chat');
      if (chat.userId !== session.user.id) {
        logWithTimestamp("Auth failed: User doesn't own this chat");
        return new Response('Unauthorized', { status: 401 });
      }
    }

    // Retrieve all previous messages for context
    logWithTimestamp('Retrieving conversation history');
    const storedMessages = await getMessagesByChatId({ id });
    logWithTimestamp('Retrieved stored messages', {
      count: storedMessages.length,
    });

    // Convert stored DB messages to UI messages format if needed
    function convertToUIMessages(messages: Array<any>): Array<UIMessage> {
      return messages.map((message) => {
        // Ensure parts is properly formatted
        const messageParts = message.parts ? message.parts : [];

        // Convert empty parts array to text part if needed (for backward compatibility)
        const formattedParts =
          Array.isArray(messageParts) && messageParts.length === 0
            ? [{ type: 'text', text: '' }]
            : messageParts;

        return {
          id: message.id,
          parts: formattedParts as UIMessage['parts'],
          role: message.role as UIMessage['role'],
          content: '', // Note: content will soon be deprecated in @ai-sdk/react
          createdAt: message.createdAt,
          experimental_attachments: (message.attachments as Array<any>) ?? [],
        };
      });
    }

    const persistedMessages = convertToUIMessages(storedMessages);

    // Use the full conversation history for context if it exists
    let contextMessages = messages;
    if (persistedMessages.length > 0) {
      logWithTimestamp('Using persisted message history for context');
      // Sort messages by createdAt to ensure proper order
      contextMessages = persistedMessages.sort((a, b) => {
        const dateA =
          a.createdAt instanceof Date
            ? a.createdAt
            : new Date(a.createdAt || 0);
        const dateB =
          b.createdAt instanceof Date
            ? b.createdAt
            : new Date(b.createdAt || 0);
        return dateA.getTime() - dateB.getTime();
      });

      // Make sure the latest user message is included
      // It might not be in the database yet since we just saved it
      const latestMessageInHistory = contextMessages.find(
        (msg) => msg.id === userMessage.id,
      );
      if (!latestMessageInHistory) {
        logWithTimestamp('Adding current user message to history context');
        contextMessages.push(userMessage);
      }

      logWithTimestamp('Context messages prepared', {
        count: contextMessages.length,
      });
    }

    logWithTimestamp('Saving user message');
    await saveMessages({
      messages: [
        {
          chatId: id,
          id: userMessage.id,
          role: 'user',
          parts: userMessage.parts,
          attachments: userMessage.experimental_attachments ?? [],
          createdAt: new Date(),
        },
      ],
    });
    logWithTimestamp('User message saved');

    // Handle Readwise integration for readwise-only and readwise-blend modes
    let readwiseContext = '';
    let enhancedMessages = contextMessages; // Start with full context instead of just current messages
    let forceNoNotesResponse = false;

    if (
      selectedChatModel === 'readwise-only' ||
      selectedChatModel === 'readwise-blend'
    ) {
      logWithTimestamp('Starting Readwise integration', {
        mode: selectedChatModel,
      });
      try {
        // Get the query from the latest user message
        const query =
          typeof userMessage.parts === 'string'
            ? userMessage.parts
            : typeof userMessage.content === 'string'
              ? userMessage.content
              : Array.isArray(userMessage.parts)
                ? userMessage.parts.join(' ')
                : JSON.stringify(userMessage.parts);

        logWithTimestamp('Processing Readwise request', { query });

        // Perform hybrid search
        logWithTimestamp('Starting hybrid search');
        const searchStartTime = Date.now();
        const matches = await hybridSearch(query);
        const searchEndTime = Date.now();
        logWithTimestamp('Hybrid search completed', {
          duration: searchEndTime - searchStartTime,
          matchCount: matches.length,
        });

        // Format the results
        logWithTimestamp('Formatting search results');
        const result = formatReadwiseContext(matches);
        logWithTimestamp('Search results formatted', {
          contextLength: result.readwiseContext.length,
          hasSources: result.hasSources,
        });

        // Handle no matches case
        if (!result.hasSources) {
          logWithTimestamp('No sources found for query');

          if (selectedChatModel === 'readwise-only') {
            logWithTimestamp(
              'Setting forceNoNotesResponse for readwise-only mode',
            );
            forceNoNotesResponse = true;
          } else {
            // readwise-blend
            logWithTimestamp(
              'No relevant notes found for readwise-blend mode - falling back to general knowledge',
            );
            readwiseContext = ''; // Empty context
            enhancedMessages = contextMessages; // Keep full conversation context
            logWithTimestamp('Using enhanced messages', {
              count: enhancedMessages.length,
            });
          }
        } else {
          logWithTimestamp('Retrieved Readwise context', {
            length: result.readwiseContext.length,
          });
          readwiseContext = result.readwiseContext;

          // Keep existing messages including full conversation history
          enhancedMessages = contextMessages;
          logWithTimestamp('Using enhanced messages with Readwise context', {
            count: enhancedMessages.length,
          });
        }
      } catch (error) {
        logWithTimestamp('Error in Readwise integration', { error });
        // Fall back to standard chat behavior on error
        if (selectedChatModel === 'readwise-only') {
          logWithTimestamp(
            'Setting forceNoNotesResponse due to error in readwise-only mode',
          );
          forceNoNotesResponse = true;
        }
      }
      logWithTimestamp('Readwise integration complete');
    }

    // Handle direct "No matching notes" response for readwise-only mode with no sources
    if (forceNoNotesResponse) {
      logWithTimestamp("Generating 'No matching notes' response");
      return createDataStreamResponse({
        execute: (dataStream) => {
          logWithTimestamp("Starting data stream for 'No matching notes'");
          const result = streamText({
            model: myProvider.languageModel('chat-model'), // Always use chat-model
            system: "Reply exactly with 'No matching notes.'",
            messages: [
              {
                role: 'user',
                parts: [
                  {
                    type: 'text',
                    text: 'Return no matching notes message',
                  },
                ],
                content: 'Return no matching notes message',
              },
            ],
            experimental_transform: smoothStream({ chunking: 'word' }),
            experimental_generateMessageId: generateUUID,
            onFinish: ({ response }) => {
              logWithTimestamp("'No matching notes' response finished", {
                messageCount: response.messages.length,
              });

              // Move DB write out of hot path
              void (async () => {
                if (session.user?.id) {
                  try {
                    logWithTimestamp(
                      "Saving 'No matching notes' response to database",
                    );
                    await saveMessages({
                      messages: [
                        {
                          id: generateUUID(),
                          chatId: id,
                          role: 'assistant',
                          parts: [
                            {
                              type: 'text',
                              text: 'No matching notes.',
                            },
                          ],
                          attachments: [],
                          createdAt: new Date(),
                        },
                      ],
                    });
                    logWithTimestamp("'No matching notes' response saved");
                  } catch (error) {
                    logWithTimestamp('Failed to save chat', { error });
                  }
                }
              })();
            },
          });

          // Pipe immediately - don't buffer tokens
          logWithTimestamp('Merging into data stream');
          result.mergeIntoDataStream(dataStream);

          // Start consuming for logging in parallel (non-blocking)
          void result.consumeStream();
        },
        onError: (error) => {
          logWithTimestamp("Error in 'No matching notes' response", { error });
          return 'No matching notes.';
        },
      });
    }

    // Select appropriate system prompt based on model
    let promptSystem = systemPrompt({ selectedChatModel });

    // Add a conversation history context guideline
    const conversationHistoryGuideline = `
CONVERSATION HISTORY GUIDELINES:
1. You have access to the COMPLETE conversation history between you and the user.
2. Always build upon previous exchanges when responding to maintain continuity.
3. If the user refers to something mentioned earlier, use that context in your response.
4. When relevant, remind the user of information they've shared previously.
5. Maintain a coherent conversation thread by referencing earlier topics or questions.
`;

    // Define enhanced versions of the prompts with proper formatting instructions
    const formattingInstructions = `
CRITICAL FORMATTING REQUIREMENT:
When including code blocks in your response:
1. NEVER place code blocks inside paragraphs.
2. ALWAYS add a blank line before and after each code block.
3. ALWAYS start code blocks with triple backticks on their own line.
4. ALWAYS end code blocks with triple backticks on their own line.
5. NEVER nest block-level elements inside paragraphs.
6. Wrap block-level content with appropriate container elements instead of <p> tags.
`;

    // Update the prompts based on model and context
    if (selectedChatModel === 'readwise-only') {
      promptSystem =
        readwiseOnlyPrompt +
        formattingInstructions +
        conversationHistoryGuideline;
    } else if (selectedChatModel === 'readwise-blend') {
      promptSystem =
        readwiseBlendPrompt +
        formattingInstructions +
        conversationHistoryGuideline;
    } else {
      promptSystem = promptSystem + conversationHistoryGuideline;
    }

    // If we have Readwise context, enhance the system prompt
    if (
      readwiseContext &&
      (selectedChatModel === 'readwise-only' ||
        selectedChatModel === 'readwise-blend')
    ) {
      // Add Readwise context to the prompt
      promptSystem = `${promptSystem}

Here is relevant information from your notes:
${readwiseContext}

REMINDER: Since you are using information from the notes above, you MUST include a "Sources" section at the end of your response using this exact format:

## Sources
- Title 1
- Title 2`;
      logWithTimestamp('Using enhanced system prompt with Readwise context', {
        promptLength: promptSystem.length,
      });
    } else if (selectedChatModel === 'readwise-blend') {
      // For readwise-blend with no relevant notes, use a prompt that allows seamless fallback
      promptSystem = `You are a helpful assistant that can answer questions based on your general knowledge.
If relevant information is provided from the user's notes, you will use that to give more personalized answers.
For this query, no relevant information was found in the user's notes, so please answer using your general knowledge.

${formattingInstructions}

Please answer the question normally without mentioning the absence of notes.`;
      logWithTimestamp(
        'Using fallback prompt for readwise-blend with no notes',
      );
    } else {
      logWithTimestamp('Using system prompt', {
        mode: selectedChatModel,
        promptLength: promptSystem.length,
      });
    }

    logWithTimestamp('Creating response stream', {
      chatMode: selectedChatModel,
      messageCount:
        selectedChatModel === 'readwise-only' ||
        selectedChatModel === 'readwise-blend'
          ? enhancedMessages.length
          : messages.length,
    });

    // Always use 'chat-model' for all cases
    const actualModelId = 'chat-model';
    logWithTimestamp('Using model', { modelId: actualModelId });

    return createDataStreamResponse({
      execute: (dataStream) => {
        logWithTimestamp('Starting data stream execution');
        const result = streamText({
          model: myProvider.languageModel(actualModelId),
          system: promptSystem,
          messages:
            selectedChatModel === 'readwise-only' ||
            selectedChatModel === 'readwise-blend'
              ? enhancedMessages
              : contextMessages,
          maxSteps: 5,
          experimental_activeTools: [], // No tools since we're simplifying to just chat-model
          experimental_transform: smoothStream({ chunking: 'word' }), // Keep 'word' chunking for compatibility
          experimental_generateMessageId: generateUUID,
          tools: {
            getWeather,
            createDocument: createDocument({ session, dataStream }),
            updateDocument: updateDocument({ session, dataStream }),
            requestSuggestions: requestSuggestions({
              session,
              dataStream,
            }),
          },
          onFinish: ({ response }) => {
            logWithTimestamp('Response stream finished', {
              totalMessages: response.messages.length,
              assistantMessages: response.messages.filter(
                (m) => m.role === 'assistant',
              ).length,
            });

            // Move DB writes out of the hot path with fire-and-forget pattern
            void (async () => {
              if (session.user?.id) {
                try {
                  logWithTimestamp('Getting trailing message ID');
                  const assistantId = getTrailingMessageId({
                    messages: response.messages.filter(
                      (message) => message.role === 'assistant',
                    ),
                  });

                  if (!assistantId) {
                    logWithTimestamp(
                      'Error: No assistant message found in response',
                    );
                    throw new Error('No assistant message found!');
                  }
                  logWithTimestamp('Found assistant message ID', {
                    id: assistantId,
                  });

                  logWithTimestamp('Appending response messages');
                  const [, assistantMessage] = appendResponseMessages({
                    messages: [userMessage],
                    responseMessages: response.messages,
                  });
                  logWithTimestamp('Response messages appended');

                  // Log parts structure for debugging
                  logWithTimestamp('Assistant message parts structure', {
                    partsType: typeof assistantMessage.parts,
                    isArray: Array.isArray(assistantMessage.parts),
                    partsLength: Array.isArray(assistantMessage.parts)
                      ? assistantMessage.parts.length
                      : 0,
                    samplePart:
                      Array.isArray(assistantMessage.parts) &&
                      assistantMessage.parts.length > 0
                        ? JSON.stringify(assistantMessage.parts[0]).substring(
                            0,
                            100,
                          )
                        : 'No parts',
                  });

                  logWithTimestamp('Saving assistant message to database');
                  await saveMessages({
                    messages: [
                      {
                        id: assistantId,
                        chatId: id,
                        role: assistantMessage.role,
                        parts: assistantMessage.parts,
                        attachments:
                          assistantMessage.experimental_attachments ?? [],
                        createdAt: new Date(),
                      },
                    ],
                  });
                  logWithTimestamp('Assistant message saved');
                } catch (error) {
                  logWithTimestamp('Failed to save chat', { error });
                }
              }
            })();
          },
          experimental_telemetry: {
            isEnabled: isProductionEnvironment,
            functionId: 'stream-text',
          },
        });

        // Pipe immediately - don't buffer tokens
        logWithTimestamp('Merging response into data stream');
        result.mergeIntoDataStream(dataStream, {
          sendReasoning: true,
        });

        // Start consuming for logging in parallel (non-blocking)
        void result.consumeStream();

        logWithTimestamp('Data stream setup complete');
      },
      onError: (error) => {
        logWithTimestamp('Error in response stream', { error });
        return 'Oops, an error occurred!';
      },
    });
  } catch (error) {
    logWithTimestamp('POST /api/chat failed', { error });
    return new Response('An error occurred while processing your request!', {
      status: 404,
    });
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  if (!id) {
    logWithTimestamp('DELETE request failed: No ID provided');
    return new Response('Not Found', { status: 404 });
  }

  const session = await auth();

  if (!session || !session.user) {
    logWithTimestamp('DELETE request failed: Unauthorized session');
    return new Response('Unauthorized', { status: 401 });
  }

  try {
    const chat = await getChatById({ id });

    if (chat.userId !== session.user.id) {
      logWithTimestamp('DELETE request failed: User does not own chat');
      return new Response('Unauthorized', { status: 401 });
    }

    logWithTimestamp('Deleting chat', { id });
    await deleteChatById({ id });
    logWithTimestamp('Chat deleted successfully');

    return new Response('Chat deleted', { status: 200 });
  } catch (error) {
    logWithTimestamp('Error deleting chat', { error });
    return new Response('An error occurred while processing your request!', {
      status: 500,
    });
  }
}
