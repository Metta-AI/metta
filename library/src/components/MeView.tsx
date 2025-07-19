'use client';

import { useState } from 'react';
import { PaperWithUserContext, User, UserInteraction } from '@/posts/data/papers';
import { toggleStarAction } from '@/posts/actions/toggleStarAction';
import { toggleQueueAction } from '@/posts/actions/toggleQueueAction';
import PaperOverlay from './PaperOverlay';
import { signOut } from 'next-auth/react';

interface MeViewProps {
  user: {
    id: string;
    name: string | null;
    email: string | null;
    image: string | null;
  };
  starredPapers: PaperWithUserContext[];
  queuedPapers: PaperWithUserContext[];
  allPapers: PaperWithUserContext[];
  users: User[];
  interactions: UserInteraction[];
}

export function MeView({ 
  user, 
  starredPapers, 
  queuedPapers, 
  allPapers, 
  users, 
  interactions 
}: MeViewProps) {
  // State for paper overlay
  const [selectedPaper, setSelectedPaper] = useState<PaperWithUserContext | null>(null);

  // Handle paper overlay close
  const handlePaperOverlayClose = () => {
    setSelectedPaper(null);
  };

  // Handle toggle star
  const handleToggleStar = async (paperId: string) => {
    try {
      const formData = new FormData();
      formData.append('paperId', paperId);
      await toggleStarAction(formData);
      
      // Update local state immediately for better UX
      setSelectedPaper(prev => {
        if (prev && prev.id === paperId) {
          return { ...prev, isStarredByCurrentUser: !prev.isStarredByCurrentUser };
        }
        return prev;
      });
    } catch (error) {
      console.error('Error toggling star:', error);
    }
  };

  // Handle toggle queue
  const handleToggleQueue = async (paperId: string) => {
    try {
      const formData = new FormData();
      formData.append('paperId', paperId);
      await toggleQueueAction(formData);
      
      // Update local state immediately for better UX
      setSelectedPaper(prev => {
        if (prev && prev.id === paperId) {
          return { ...prev, isQueuedByCurrentUser: !prev.isQueuedByCurrentUser };
        }
        return prev;
      });
    } catch (error) {
      console.error('Error toggling queue:', error);
    }
  };

  // Handle paper click
  const handlePaperClick = (paper: PaperWithUserContext) => {
    setSelectedPaper(paper);
  };

  // Handle remove from starred
  const handleRemoveFromStarred = async (paperId: string) => {
    await handleToggleStar(paperId);
  };

  // Handle remove from queued
  const handleRemoveFromQueued = async (paperId: string) => {
    await handleToggleQueue(paperId);
  };

  // Handle sign out
  const handleSignOut = async () => {
    try {
      await signOut({ callbackUrl: '/' });
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  // Generate user initials for profile circle
  const getUserInitials = (name: string | null, email: string | null) => {
    if (name) {
      return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
    }
    if (email) {
      return email.charAt(0).toUpperCase();
    }
    return '?';
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* User Profile Section */}
      <div className="mb-8">
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center gap-4">
            {/* Profile Circle */}
            <div className="w-16 h-16 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-semibold flex-shrink-0">
              {getUserInitials(user.name, user.email)}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-700">Name:</span>
                <span className="text-lg font-semibold text-gray-900">
                  {user.name || 'Unknown User'}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-sm font-medium text-gray-700">Email:</span>
                <span className="text-lg text-gray-900">{user.email}</span>
              </div>
            </div>
            <button
              onClick={handleSignOut}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm font-medium"
            >
              Sign Out
            </button>
          </div>
        </div>
      </div>

      {/* Starred Papers Section */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-gray-900 mb-3">
          Starred ({starredPapers.length})
        </h2>
        {starredPapers.length > 0 ? (
          <div className="bg-white rounded-lg border border-gray-200 divide-y divide-gray-200">
            {starredPapers.map(paper => (
              <div key={paper.id} className="p-3 hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <button
                      onClick={() => handlePaperClick(paper)}
                      className="text-left w-full hover:text-blue-600 transition-colors"
                    >
                      <h3 className="text-base font-medium text-gray-900 truncate">
                        {paper.title}
                      </h3>
                      {paper.authors && paper.authors.length > 0 && (
                        <p className="text-sm text-gray-600 mt-1">
                          {paper.authors.join(', ')}
                        </p>
                      )}
                    </button>
                  </div>
                  <button
                    onClick={() => handleRemoveFromStarred(paper.id)}
                    className="ml-3 p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="Remove from starred"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-lg border border-gray-200 p-4 text-center">
            <p className="text-gray-500">No starred papers yet.</p>
          </div>
        )}
      </div>

      {/* Queued Papers Section */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-gray-900 mb-3">
          Queued ({queuedPapers.length})
        </h2>
        {queuedPapers.length > 0 ? (
          <div className="bg-white rounded-lg border border-gray-200 divide-y divide-gray-200">
            {queuedPapers.map(paper => (
              <div key={paper.id} className="p-3 hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <button
                      onClick={() => handlePaperClick(paper)}
                      className="text-left w-full hover:text-blue-600 transition-colors"
                    >
                      <h3 className="text-base font-medium text-gray-900 truncate">
                        {paper.title}
                      </h3>
                      {paper.authors && paper.authors.length > 0 && (
                        <p className="text-sm text-gray-600 mt-1">
                          {paper.authors.join(', ')}
                        </p>
                      )}
                    </button>
                  </div>
                  <button
                    onClick={() => handleRemoveFromQueued(paper.id)}
                    className="ml-3 p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="Remove from queue"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-lg border border-gray-200 p-4 text-center">
            <p className="text-gray-500">No queued papers yet.</p>
          </div>
        )}
      </div>

      {/* Paper Overlay */}
      {selectedPaper && (
        <PaperOverlay
          paper={selectedPaper}
          users={users}
          interactions={interactions}
          onClose={handlePaperOverlayClose}
          onStarToggle={handleToggleStar}
          onQueueToggle={handleToggleQueue}
        />
      )}
    </div>
  );
} 