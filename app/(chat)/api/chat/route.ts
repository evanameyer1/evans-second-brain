import {
  UIMessage,
  appendResponseMessages,
  createDataStreamResponse,
  smoothStream,
  streamText,
} from 'ai';
import { auth } from '@/app/(auth)/auth';
import { systemPrompt, readwiseOnlyPrompt, readwiseBlendPrompt } from '@/lib/ai/prompts';
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

export async function POST(request: Request) {
  console.log("===== CHAT POST REQUEST STARTED =====");
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

    console.log(`Chat ID: ${id}`);
    console.log(`Selected model: ${selectedChatModel}`);
    console.log(`Total messages: ${messages.length}`);

    const session = await auth();

    if (!session || !session.user || !session.user.id) {
      console.log("Auth failed: Unauthorized session");
      return new Response('Unauthorized', { status: 401 });
    }

    console.log(`Authenticated user: ${session.user.id}`);
    const userMessage = getMostRecentUserMessage(messages);

    if (!userMessage) {
      console.log("Error: No user message found in request");
      return new Response('No user message found', { status: 400 });
    }

    // Log user message with safe JSON stringify
    console.log("Latest user message:", JSON.stringify(userMessage).substring(0, 200) + "...");
    
    const chat = await getChatById({ id });

    if (!chat) {
      console.log("Creating new chat...");
      const title = await generateTitleFromUserMessage({
        message: userMessage,
      });
      console.log(`Generated title: "${title}"`);

      await saveChat({ id, userId: session.user.id, title });
      console.log("New chat saved successfully");
    } else {
      console.log("Using existing chat");
      if (chat.userId !== session.user.id) {
        console.log("Auth failed: User doesn't own this chat");
        return new Response('Unauthorized', { status: 401 });
      }
    }

    // Retrieve all previous messages for context
    console.log("Retrieving conversation history...");
    const storedMessages = await getMessagesByChatId({ id });
    console.log(`Retrieved ${storedMessages.length} stored messages for context`);
    
    // Convert stored DB messages to UI messages format if needed
    function convertToUIMessages(messages: Array<any>): Array<UIMessage> {
      return messages.map((message) => ({
        id: message.id,
        parts: message.parts as UIMessage['parts'],
        role: message.role as UIMessage['role'],
        content: '', // Note: content will soon be deprecated in @ai-sdk/react
        createdAt: message.createdAt,
        experimental_attachments: (message.attachments as Array<any>) ?? [],
      }));
    }
    
    const persistedMessages = convertToUIMessages(storedMessages);
    
    // Use the full conversation history for context if it exists
    let contextMessages = messages;
    if (persistedMessages.length > 0) {
      console.log("Using persisted message history for context");
      // Sort messages by createdAt to ensure proper order
      contextMessages = persistedMessages.sort((a, b) => {
        const dateA = a.createdAt instanceof Date ? a.createdAt : new Date(a.createdAt || 0);
        const dateB = b.createdAt instanceof Date ? b.createdAt : new Date(b.createdAt || 0);
        return dateA.getTime() - dateB.getTime();
      });
      
      // Make sure the latest user message is included
      // It might not be in the database yet since we just saved it
      const latestMessageInHistory = contextMessages.find(msg => msg.id === userMessage.id);
      if (!latestMessageInHistory) {
        console.log("Adding current user message to history context");
        contextMessages.push(userMessage);
      }
      
      console.log(`Using ${contextMessages.length} messages for full conversation context`);
    }

    console.log("Saving user message...");
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
    console.log("User message saved successfully");

    // Handle Readwise integration for readwise-only and readwise-blend modes
    let readwiseContext = '';
    let enhancedMessages = contextMessages; // Start with full context instead of just current messages
    let forceNoNotesResponse = false;
    
    if (selectedChatModel === 'readwise-only' || selectedChatModel === 'readwise-blend') {
      console.log(`\n===== STARTING READWISE INTEGRATION (${selectedChatModel} mode) =====`);
      try {
        // Get the query from the latest user message
        const query = typeof userMessage.parts === 'string' 
          ? userMessage.parts 
          : typeof userMessage.content === 'string'
            ? userMessage.content
            : Array.isArray(userMessage.parts) 
              ? userMessage.parts.join(' ')
              : JSON.stringify(userMessage.parts);
        
        console.log(`Processing ${selectedChatModel} request with query: "${query}"`);
        
        // Perform hybrid search
        console.log("Calling hybridSearch...");
        const searchStartTime = Date.now();
        const matches = await hybridSearch(query);
        const searchEndTime = Date.now();
        console.log(`hybridSearch completed in ${searchEndTime - searchStartTime}ms`);
        
        // Log summary of search results
        console.log(`\n===== SEARCH RESULTS SUMMARY =====`);
        console.log(`Query: "${query}"`);
        console.log(`Model: ${selectedChatModel}`);
        console.log(`Total matches found: ${matches.length}`);
        if (matches.length > 0) {
          console.log(`Top match score: ${matches[0].score.toFixed(4)}`);
          console.log(`Top match title: "${matches[0].title}"`);
        }
        console.log(`================================\n`);
        
        // Format the results
        console.log("Formatting search results...");
        const result = formatReadwiseContext(matches);
        console.log(`Format complete, context length: ${result.readwiseContext.length} chars`);
        console.log(`Has sources: ${result.hasSources}`);
        
        // Handle no matches case
        if (!result.hasSources) {
          console.log(`No sources found for query: "${query}"`);
          
          if (selectedChatModel === 'readwise-only') {
            console.log("Setting forceNoNotesResponse to true for readwise-only mode");
            forceNoNotesResponse = true;
          } else { // readwise-blend
            console.log("No relevant notes found for readwise-blend mode - falling back to general knowledge");
            // Don't add any special message - let the model use its general knowledge
            readwiseContext = ""; // Empty context
            enhancedMessages = contextMessages; // Keep full conversation context
            console.log(`Using ${enhancedMessages.length} messages with enhanced system prompt`);
          }
        } else {
          console.log(`Retrieved Readwise context (${result.readwiseContext.length} chars)`);
          readwiseContext = result.readwiseContext;
          
          // Will be combined with the selected prompt later in the code
          console.log(`Readwise context retrieved (${readwiseContext.length} chars)`);
          
          // Keep existing messages including full conversation history
          enhancedMessages = contextMessages;
          console.log(`Using ${enhancedMessages.length} messages with enhanced system prompt`);
        }
      } catch (error) {
        console.error("Error in Readwise integration:", error);
        console.log("Stack trace:", error instanceof Error ? error.stack : "No stack available");
        // Fall back to standard chat behavior on error
        if (selectedChatModel === 'readwise-only') {
          console.log("Setting forceNoNotesResponse to true due to error in readwise-only mode");
          forceNoNotesResponse = true;
        }
      }
      console.log(`===== READWISE INTEGRATION COMPLETE =====\n`);
    }

    // Handle direct "No matching notes" response for readwise-only mode with no sources
    if (forceNoNotesResponse) {
      console.log("Generating 'No matching notes' response...");
      return createDataStreamResponse({
        execute: (dataStream) => {
          console.log("Starting data stream for 'No matching notes'");
          const result = streamText({
            model: myProvider.languageModel('chat-model'), // Always use chat-model
            system: "Reply exactly with 'No matching notes.'",
            messages: [
              {
                role: 'user',
                parts: [{
                  type: 'text',
                  text: 'Return no matching notes message'
                }],
                content: 'Return no matching notes message',
              }
            ],
            experimental_transform: smoothStream({ chunking: 'word' }),
            experimental_generateMessageId: generateUUID,
            onFinish: async ({ response }) => {
              console.log("'No matching notes' response finished");
              console.log("Response messages:", response.messages.length);
              if (session.user?.id) {
                try {
                  console.log("Saving 'No matching notes' response to database");
                  await saveMessages({
                    messages: [
                      {
                        id: generateUUID(),
                        chatId: id,
                        role: 'assistant',
                        parts: [{
                          type: 'text',
                          text: "No matching notes."
                        }],
                        attachments: [],
                        createdAt: new Date(),
                      },
                    ],
                  });
                  console.log("'No matching notes' response saved successfully");
                } catch (error) {
                  console.error('Failed to save chat:', error);
                }
              }
            },
          });

          console.log("Consuming stream and merging into data stream");
          result.consumeStream();
          result.mergeIntoDataStream(dataStream);
        },
        onError: (error) => {
          console.error("Error in 'No matching notes' response:", error);
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
      promptSystem = readwiseOnlyPrompt + formattingInstructions + conversationHistoryGuideline;
    } else if (selectedChatModel === 'readwise-blend') {
      promptSystem = readwiseBlendPrompt + formattingInstructions + conversationHistoryGuideline;
    } else {
      promptSystem = promptSystem + conversationHistoryGuideline;
    }
    
    // If we have Readwise context, enhance the system prompt
    if (readwiseContext && (selectedChatModel === 'readwise-only' || selectedChatModel === 'readwise-blend')) {
      // Add Readwise context to the prompt
      promptSystem = `${promptSystem}

Here is relevant information from your notes:
${readwiseContext}

REMINDER: Since you are using information from the notes above, you MUST include a "Sources" section at the end of your response using this exact format:

## Sources
- Title 1
- Title 2`;
      console.log(`Using enhanced system prompt with Readwise context (${promptSystem.length} chars)`);
    } else if (selectedChatModel === 'readwise-blend') {
      // For readwise-blend with no relevant notes, use a prompt that allows seamless fallback
      promptSystem = `You are a helpful assistant that can answer questions based on your general knowledge.
If relevant information is provided from the user's notes, you will use that to give more personalized answers.
For this query, no relevant information was found in the user's notes, so please answer using your general knowledge.

${formattingInstructions}

Please answer the question normally without mentioning the absence of notes.`;
      console.log(`Using fallback prompt for readwise-blend with no notes`);
    } else {
      console.log(`Using system prompt for mode: ${selectedChatModel}`);
      console.log(`System prompt length: ${promptSystem.length} chars`);
    }

    console.log("\n===== CREATING RESPONSE STREAM =====");
    console.log(`Selected chat mode: ${selectedChatModel}`);
    console.log(`Total messages to send: ${(selectedChatModel === 'readwise-only' || selectedChatModel === 'readwise-blend') ? enhancedMessages.length : messages.length}`);
    
    // Always use 'chat-model' for all cases
    const actualModelId = 'chat-model';
    console.log(`Using model: ${actualModelId} for all modes`);
    
    return createDataStreamResponse({
      execute: (dataStream) => {
        console.log("Starting data stream execution");
        const result = streamText({
          model: myProvider.languageModel(actualModelId),
          system: promptSystem,
          messages: selectedChatModel === 'readwise-only' || selectedChatModel === 'readwise-blend' 
            ? enhancedMessages 
            : contextMessages,
          maxSteps: 5,
          experimental_activeTools: [], // No tools since we're simplifying to just chat-model
          experimental_transform: smoothStream({ chunking: 'word' }),
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
          onFinish: async ({ response }) => {
            console.log("Response stream finished");
            console.log(`Total response messages: ${response.messages.length}`);
            console.log(`Assistant messages: ${response.messages.filter(m => m.role === 'assistant').length}`);
            
            if (session.user?.id) {
              try {
                console.log("Getting trailing message ID");
                const assistantId = getTrailingMessageId({
                  messages: response.messages.filter(
                    (message) => message.role === 'assistant',
                  ),
                });

                if (!assistantId) {
                  console.error("No assistant message found in response!");
                  throw new Error('No assistant message found!');
                }
                console.log(`Assistant message ID: ${assistantId}`);

                console.log("Appending response messages");
                const [, assistantMessage] = appendResponseMessages({
                  messages: [userMessage],
                  responseMessages: response.messages,
                });
                console.log("Response messages appended successfully");

                console.log("Saving assistant message to database");
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
                console.log("Assistant message saved successfully");
              } catch (error) {
                console.error('Failed to save chat:', error);
                console.error('Error details:', error instanceof Error ? error.message : error);
              }
            }
          },
          experimental_telemetry: {
            isEnabled: isProductionEnvironment,
            functionId: 'stream-text',
          },
        });

        console.log("Consuming response stream");
        result.consumeStream();

        console.log("Merging response into data stream");
        result.mergeIntoDataStream(dataStream, {
          sendReasoning: true,
        });
        console.log("Data stream merge complete");
      },
      onError: (error) => {
        console.error("Error in response stream:", error);
        console.error("Error details:", error instanceof Error ? error.message : error);
        return 'Oops, an error occurred!';
      },
    });
  } catch (error) {
    console.error('POST /api/chat failed:', error);
    console.error('Error stack:', error instanceof Error ? error.stack : "No stack available");
    return new Response('An error occurred while processing your request!', {
      status: 404,
    });
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  if (!id) {
    return new Response('Not Found', { status: 404 });
  }

  const session = await auth();

  if (!session || !session.user) {
    return new Response('Unauthorized', { status: 401 });
  }

  try {
    const chat = await getChatById({ id });

    if (chat.userId !== session.user.id) {
      return new Response('Unauthorized', { status: 401 });
    }

    await deleteChatById({ id });

    return new Response('Chat deleted', { status: 200 });
  } catch (error) {
    return new Response('An error occurred while processing your request!', {
      status: 500,
    });
  }
}
