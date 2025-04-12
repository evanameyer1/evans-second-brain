'use client';

import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { useState } from 'react';

export default function ReadwisePage() {
  const [isSyncing, setIsSyncing] = useState(false);

  const handleSync = async () => {
    setIsSyncing(true);
    try {
      const response = await fetch('/api/readwise/sync', {
        method: 'POST',
      });
      
      const data = await response.json();
      
      if (response.ok) {
        toast.success('Readwise library synced ✅');
      } else {
        toast.error(`Failed to sync: ${data.message || 'Unknown error'}`);
      }
    } catch (error) {
      toast.error('Failed to sync Readwise library');
      console.error('Error syncing Readwise:', error);
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-2xl font-bold mb-6">Readwise Integration</h1>
      
      <div className="max-w-lg w-full bg-muted p-6 rounded-lg shadow-sm">
        <h2 className="text-xl font-semibold mb-4">Sync Your Second Brain</h2>
        <p className="mb-6 text-muted-foreground">
          Click the button below to sync your Readwise highlights to your second brain.
          This process may take a few minutes depending on the size of your library.
        </p>
        
        <Button 
          onClick={handleSync} 
          disabled={isSyncing}
          className="w-full"
        >
          {isSyncing ? 'Syncing...' : 'Sync Readwise Library'}
        </Button>
      </div>
    </div>
  );
} 