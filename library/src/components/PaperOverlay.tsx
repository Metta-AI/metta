'use client';

import React from 'react';
import { PaperWithUserContext, User, UserInteraction } from '@/posts/data/papers';

interface PaperOverlayProps {
  paper: PaperWithUserContext;
  users: User[];
  interactions: UserInteraction[];
  onClose: () => void;
  onStarToggle: (paperId: string) => void;
  onQueueToggle: (paperId: string) => void;
}

export default function PaperOverlay({
  paper,
  users,
  interactions,
  onClose,
  onStarToggle,
  onQueueToggle,
}: PaperOverlayProps) {
  // Get interactions for this paper
  const paperInteractions = interactions.filter(i => i.paperId === paper.id);
  
  // Get users who have interacted with this paper
  const usersWithInteractions = users.filter(user => 
    paperInteractions.some(interaction => interaction.userId === user.id)
  );

  // Get users by interaction type
  const starredUsers = usersWithInteractions.filter(user =>
    paperInteractions.some(i => i.userId === user.id && i.starred)
  );
  const queuedUsers = usersWithInteractions.filter(user =>
    paperInteractions.some(i => i.userId === user.id && i.queued)
  );
  const readUsers = usersWithInteractions.filter(user =>
    paperInteractions.some(i => i.userId === user.id && i.readAt)
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Semi-transparent backdrop */}
      <div 
        className="absolute inset-0 bg-black/20 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Paper overlay card */}
      <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto bg-white rounded-xl shadow-2xl">
        {/* Content */}
        <div className="p-6 space-y-4">
          {/* Title with star toggle and close button */}
          <div className="flex items-start gap-3">
            <button 
              onClick={() => onStarToggle(paper.id)} 
              className="focus:outline-none hover:scale-110 transition-transform flex-shrink-0 mt-1"
              aria-label={paper.isStarredByCurrentUser ? 'Remove from favorites' : 'Add to favorites'}
            >
              {paper.isStarredByCurrentUser ? (
                <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/>
                </svg>
              ) : (
                <svg className="w-5 h-5 text-gray-300 hover:text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 20 20">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"/>
                </svg>
              )}
            </button>
            <h1 className="text-2xl font-bold text-gray-900 leading-tight flex-1">
              {paper.title}
            </h1>
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors flex-shrink-0 mt-1"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Metadata grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Authors */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Authors</h3>
              <p className="text-gray-900">
                {paper.authors && paper.authors.length > 0 ? paper.authors.map(author => author.name).join(', ') : ''}
              </p>
            </div>

            {/* Institutions */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Institutions</h3>
              <p className="text-gray-900">
                {paper.institutions && paper.institutions.length > 0 ? paper.institutions.join(', ') : ''}
              </p>
            </div>
          </div>

          {/* Abstract */}
          {paper.abstract && (
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-3">Abstract</h3>
              <p className="text-gray-900 leading-relaxed text-lg">
                {paper.abstract}
              </p>
            </div>
          )}

          {/* Topic tags */}
          {paper.tags && paper.tags.length > 0 && (
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-medium text-gray-700">Topics</h3>
                <div className="flex flex-wrap gap-2">
                  {paper.tags.map((topic: string, index: number) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full font-medium"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* URL */}
          {paper.link && (
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-medium text-gray-700">URL</h3>
                <a
                  href={paper.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 underline break-all"
                >
                  {paper.link}
                </a>
              </div>
            </div>
          )}

          {/* User interactions */}
          <div className="space-y-4">
            {/* Starred by */}
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-medium text-gray-700">Starred by</h3>
                {starredUsers.length > 0 && (
                  <>
                    <div className="relative inline-flex items-center justify-center">
                      <svg className="w-8 h-8 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/>
                      </svg>
                      <span className="absolute text-sm font-medium text-black">
                        {starredUsers.length}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {starredUsers.map(user => (
                        <span
                          key={user.id}
                          className="inline-flex items-center px-3 py-1 bg-yellow-100 text-yellow-800 text-sm rounded-full font-medium"
                        >
                          {user.name || user.email}
                        </span>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Read by */}
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-medium text-gray-700">Read by</h3>
                <div className="flex flex-wrap gap-2">
                  {readUsers.map(user => (
                    <span
                      key={user.id}
                      className="inline-flex items-center px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full font-medium"
                    >
                      {user.name || user.email}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Queued by with add to queue button */}
            <div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <h3 className="text-sm font-medium text-gray-700">Queued by</h3>
                  <div className="flex flex-wrap gap-2">
                    {queuedUsers.map(user => (
                      <span
                        key={user.id}
                        className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full font-medium"
                      >
                        {user.name || user.email}
                      </span>
                    ))}
                  </div>
                </div>
                <button
                  onClick={() => onQueueToggle(paper.id)}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    paper.isQueuedByCurrentUser
                      ? 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {paper.isQueuedByCurrentUser ? 'Remove from queue' : 'Add to queue'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 