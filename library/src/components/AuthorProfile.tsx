"use client";

import { FC, useState } from "react";

import { AuthorDTO } from "@/posts/data/authors-client";

interface AuthorProfileProps {
  author: AuthorDTO;
  onClose?: () => void;
}

/**
 * AuthorProfile Component
 * 
 * Displays detailed information about a single author including their profile,
 * statistics, papers, and network information in a tabbed interface.
 * Designed to work as an overlay/modal.
 */
export const AuthorProfile: FC<AuthorProfileProps> = ({ author, onClose }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'papers' | 'network'>('overview');
  const [isFollowing, setIsFollowing] = useState(false);

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const formatDate = (date: Date | string) => {
    // Convert string to Date if needed
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    
    // Check if the date is valid
    if (isNaN(dateObj.getTime())) {
      return 'Invalid date';
    }
    
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(dateObj);
  };

  const formatRelativeDate = (date: Date | string | null) => {
    if (!date) return 'Unknown';
    
    // Convert string to Date if needed
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    
    // Check if the date is valid
    if (isNaN(dateObj.getTime())) {
      return 'Unknown';
    }
    
    const now = new Date();
    const diffInDays = Math.floor((now.getTime() - dateObj.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffInDays === 0) return 'Today';
    if (diffInDays === 1) return 'Yesterday';
    if (diffInDays < 7) return `${diffInDays} days ago`;
    if (diffInDays < 30) return `${Math.floor(diffInDays / 7)} weeks ago`;
    return `${Math.floor(diffInDays / 30)} months ago`;
  };

  const toggleFollow = () => {
    setIsFollowing(!isFollowing);
  };

  return (
    <div className="bg-white">
      {/* Header */}
      <div className="border-b border-gray-200">
        <div className="px-6 py-6">
          {/* Close Button */}
          {onClose && (
            <div className="flex justify-end mb-4">
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                aria-label="Close"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}

          {/* Author Info */}
          <div className="flex items-start gap-6">
            <div className="w-24 h-24 bg-primary-500 text-white rounded-full flex items-center justify-center text-3xl font-semibold flex-shrink-0">
              {author.avatar || getInitials(author.name)}
            </div>
            <div className="flex-1 min-w-0">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">{author.name}</h2>
              {author.title && (
                <p className="text-xl text-gray-600 mb-1">{author.title}</p>
              )}
              {author.institution && (
                <p className="text-lg text-gray-500 mb-3">{author.institution}</p>
              )}
              
              <div className="flex items-center gap-4 mb-4">
                <div className="flex items-center gap-6 text-sm text-gray-600">
                  <div>
                    <span className="font-semibold text-gray-900 text-lg">{author.hIndex || 0}</span>
                    <span className="ml-1">h-index</span>
                  </div>
                  <div>
                    <span className="font-semibold text-gray-900 text-lg">
                      {(author.totalCitations || 0).toLocaleString()}
                    </span>
                    <span className="ml-1">citations</span>
                  </div>
                  <div>
                    <span className="font-semibold text-gray-900 text-lg">{author.paperCount}</span>
                    <span className="ml-1">papers</span>
                  </div>
                </div>
                <button
                  onClick={toggleFollow}
                  className={`px-6 py-2 rounded-full font-medium transition-colors ${
                    isFollowing
                      ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      : 'bg-primary-500 text-white hover:bg-primary-600'
                  }`}
                >
                  {isFollowing ? 'Following' : 'Follow'}
                </button>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-semibold ${
                    author.claimed
                      ? 'bg-green-100 text-green-700 border border-green-200'
                      : 'bg-gray-100 text-gray-600 border border-gray-200'
                  }`}
                >
                  {author.claimed ? 'Claimed Profile' : 'Unclaimed Profile'}
                </span>
              </div>

              {author.expertise.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {author.expertise.map((exp, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded-full"
                    >
                      {exp}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="px-6">
          <div className="flex space-x-8">
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'papers', label: `Papers (${author.recentPapers.length})` },
              { id: 'network', label: 'Network' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 py-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Papers</h3>
                <div className="space-y-4">
                  {author.recentPapers.slice(0, 5).map((paper) => (
                    <div key={paper.id} className="border-b border-gray-100 pb-4 last:border-b-0">
                      <h4 className="font-medium text-gray-900 mb-1">{paper.title}</h4>
                      <p className="text-sm text-gray-600 mb-2">
                        {formatDate(paper.createdAt)} • {paper.stars} stars
                      </p>
                      {paper.link && (
                        <a
                          href={paper.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-primary-500 hover:text-primary-600 underline"
                        >
                          View Paper
                        </a>
                      )}
                    </div>
                  ))}
                  {author.recentPapers.length === 0 && (
                    <p className="text-gray-500 text-sm">No papers found for this author.</p>
                  )}
                </div>
              </div>
            </div>
            <div className="space-y-6">
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Activity</h3>
                <p className="text-sm text-gray-600">
                  Last active {formatRelativeDate(author.recentActivity)}
                </p>
              </div>
              
              {author.orcid && (
                <div className="bg-white rounded-lg border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">External Profiles</h3>
                  <div className="space-y-2">
                    <a
                      href={`https://orcid.org/${author.orcid}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-primary-500 hover:text-primary-600 underline block"
                    >
                      ORCID Profile
                    </a>
                    {author.googleScholarId && (
                      <a
                        href={`https://scholar.google.com/citations?user=${author.googleScholarId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-primary-500 hover:text-primary-600 underline block"
                      >
                        Google Scholar
                      </a>
                    )}
                    {author.arxivId && (
                      <a
                        href={`https://arxiv.org/a/${author.arxivId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-primary-500 hover:text-primary-600 underline block"
                      >
                        arXiv Profile
                      </a>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'papers' && (
          <div className="bg-white rounded-lg border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">All Papers</h3>
            </div>
            <div className="divide-y divide-gray-200">
              {author.recentPapers.map((paper) => (
                <div key={paper.id} className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="text-lg font-medium text-gray-900 mb-2">{paper.title}</h4>
                      <div className="flex items-center gap-4 text-sm text-gray-600 mb-3">
                        <span>{formatDate(paper.createdAt)}</span>
                        <span>•</span>
                        <span>{paper.stars} stars</span>
                      </div>
                    </div>
                    {paper.link && (
                      <a
                        href={paper.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-4 text-primary-500 hover:text-primary-600"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 3h7v7m0 0L10 21l-7-7 11-11z" />
                        </svg>
                      </a>
                    )}
                  </div>
                </div>
              ))}
              {author.recentPapers.length === 0 && (
                <div className="p-6 text-center text-gray-500">
                  No papers found for this author.
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'network' && (
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Collaboration Network</h3>
            <p className="text-gray-600">Network visualization coming soon...</p>
          </div>
        )}
      </div>
    </div>
  );
}; 