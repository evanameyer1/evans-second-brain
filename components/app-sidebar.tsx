'use client';

import type { User } from 'next-auth';
import { useRouter } from 'next/navigation';

import { PlusIcon, ReadwiseIcon } from '@/components/icons';
import { SidebarHistory } from '@/components/sidebar-history';
import { SidebarUserNav } from '@/components/sidebar-user-nav';
import { Button } from '@/components/ui/button';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  useSidebar,
} from '@/components/ui/sidebar';
import Link from 'next/link';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';
import { toast } from 'sonner';
import { useState } from 'react';

export function AppSidebar({ user }: { user: User | undefined }) {
  const router = useRouter();
  const { setOpenMobile } = useSidebar();
  const [isSyncing, setIsSyncing] = useState(false);

  const handleReadwiseSync = async () => {
    setIsSyncing(true);
    try {
      const response = await fetch('/api/readwise/sync', {
        method: 'POST',
      });
      
      const data = await response.json();
      
      if (response.ok) {
        toast.success('Readwise Reader documents synced ✅');
      } else {
        toast.error(`Failed to sync: ${data.message || 'Unknown error'}`);
      }
    } catch (error) {
      toast.error('Failed to sync Readwise Reader documents');
      console.error('Error syncing Readwise Reader:', error);
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <Sidebar className="group-data-[side=left]:border-r-0">
      <SidebarHeader>
        <SidebarMenu>
          <div className="flex flex-row justify-between items-center">
            <Link
              href="/"
              onClick={() => {
                setOpenMobile(false);
              }}
              className="flex flex-row gap-3 items-center"
            >
              <span className="text-lg font-semibold px-2 hover:bg-muted rounded-md cursor-pointer">
                Evan&apos;s Second Brain
              </span>
            </Link>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  type="button"
                  className="p-2 h-fit"
                  onClick={() => {
                    setOpenMobile(false);
                    router.push('/');
                    router.refresh();
                  }}
                >
                  <PlusIcon />
                </Button>
              </TooltipTrigger>
              <TooltipContent align="end">New Chat</TooltipContent>
            </Tooltip>
          </div>
        </SidebarMenu>
      </SidebarHeader>

      <div className="px-3 pt-3 pb-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              type="button"
              className="w-full"
              size="sm"
              onClick={handleReadwiseSync}
              disabled={isSyncing}
            >
              <ReadwiseIcon />
              <span className="ml-2">{isSyncing ? 'Syncing...' : 'Sync Reader'}</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent align="center">Sync documents from Readwise Reader to your Second Brain</TooltipContent>
        </Tooltip>
      </div>

      <SidebarContent>
        <SidebarHistory user={user} />
      </SidebarContent>
      <SidebarFooter>{user && <SidebarUserNav user={user} />}</SidebarFooter>
    </Sidebar>
  );
}
