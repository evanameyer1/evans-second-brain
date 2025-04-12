import { syncReadwise } from '@/lib/readwiseSync';
import { NextResponse } from 'next/server';

export async function POST() {
  try {
    console.log("API route: Readwise Reader sync initiated");
    
    // Validate required environment variables
    if (!process.env.READWISE_TOKEN) {
      console.error("READWISE_TOKEN is missing");
      return NextResponse.json(
        { success: false, message: 'READWISE_TOKEN is not configured' },
        { status: 500 }
      );
    }

    if (!process.env.PINECONE_API_KEY) {
      console.error("Pinecone configuration is missing");
      return NextResponse.json(
        { success: false, message: 'Pinecone is not properly configured' },
        { status: 500 }
      );
    }
    
    // Get the last sync time from query params if available
    const lastSync = process.env.LAST_SYNC_TIME || undefined;
    
    await syncReadwise(lastSync);
    
    // Update the last sync time
    const now = new Date().toISOString();
    console.log(`API route: Readwise Reader sync completed successfully at ${now}`);
    
    return NextResponse.json({ 
      success: true, 
      message: 'Readwise Reader documents synced successfully',
      syncTime: now 
    });
  } catch (error) {
    console.error('Error syncing Readwise Reader:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    
    return NextResponse.json(
      { 
        success: false, 
        message: 'Failed to sync Readwise Reader documents', 
        error: errorMessage
      },
      { status: 500 }
    );
  }
} 