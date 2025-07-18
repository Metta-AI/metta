'use client';

import { useState } from 'react';
import { PaperWithUserContext, User, UserInteraction } from '@/posts/data/papers';
import { toggleStarAction } from '@/posts/actions/toggleStarAction';
import { toggleQueueAction } from '@/posts/actions/toggleQueueAction';
import PaperOverlay from './PaperOverlay';

interface UserCardProps {
  user: User;
  allPapers: PaperWithUserContext[];
  users: User[];
  interactions: UserInteraction[];
  onClose: () => void;
}

export default function UserCard({
  user,
  allPapers,
  users,
  interactions,
  onClose,
}: UserCardProps) {
  // State for paper overlay
  const [selectedPaper, setSelectedPaper] = useState<PaperWithUserContext | null>(null);

  // Get papers starred by this user
  const starredPapers = allPapers.filter(paper => {
    const userInteraction = interactions.find(i => 
      i.userId === user.id && i.paperId === paper.id && i.starred
    );
    return userInteraction !== undefined;
  });

  // Get papers queued by this user
  const queuedPapers = allPapers.filter(paper => {
    const userInteraction = interactions.find(i => 
      i.userId === user.id && i.paperId === paper.id && i.queued
    );
    return userInteraction !== undefined;
  });

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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Semi-transparent backdrop */}
      <div 
        className="absolute inset-0 bg-black/20 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* User card */}
      <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto bg-white rounded-xl shadow-2xl">
        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Header with close button */}
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              {/* Profile Circle */}
              <div className="w-16 h-16 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-semibold flex-shrink-0">
                {getUserInitials(user.name, user.email)}
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  {user.name || 'Unknown User'}
                </h1>
                <p className="text-lg text-gray-600">{user.email}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors flex-shrink-0"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Starred Papers Section */}
          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              Starred ({starredPapers.length})
            </h2>
            {starredPapers.length > 0 ? (
              <div className="bg-gray-50 rounded-lg border border-gray-200 divide-y divide-gray-200">
                {starredPapers.map(paper => (
                  <div key={paper.id} className="p-3 hover:bg-gray-100">
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
                ))}
              </div>
            ) : (
              <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 text-center">
                <p className="text-gray-500">No starred papers yet.</p>
              </div>
            )}
          </div>

          {/* Queued Papers Section */}
          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              Queued ({queuedPapers.length})
            </h2>
            {queuedPapers.length > 0 ? (
              <div className="bg-gray-50 rounded-lg border border-gray-200 divide-y divide-gray-200">
                {queuedPapers.map(paper => (
                  <div key={paper.id} className="p-3 hover:bg-gray-100">
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
                ))}
              </div>
            ) : (
              <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 text-center">
                <p className="text-gray-500">No queued papers yet.</p>
              </div>
            )}
          </div>
        </div>
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