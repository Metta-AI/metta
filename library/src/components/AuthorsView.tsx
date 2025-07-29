"use client";

import React, { FC, useState, useEffect, useRef, useMemo } from "react";

import { AuthorDTO, loadAuthorClient } from "@/posts/data/authors-client";
import { AuthorProfile } from "./AuthorProfile";

interface AuthorsViewProps {
  authors: AuthorDTO[];
}

/**
 * AuthorsView Component
 * 
 * Displays a grid of author cards with filtering and sorting capabilities.
 * Matches the mockup design exactly with filter controls and no header.
 */
export const AuthorsView: FC<AuthorsViewProps> = ({ authors }) => {
  const [selectedAuthor, setSelectedAuthor] = useState<AuthorDTO | null>(null);
  const [isOverlayOpen, setIsOverlayOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('name');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [authorsList, setAuthorsList] = useState<AuthorDTO[]>(authors);
  const filterInputRef = useRef<HTMLInputElement>(null);

  // Update authors list when props change
  useEffect(() => {
    setAuthorsList(authors);
  }, [authors]);

  // Filter and sort authors based on current state
  const filteredAndSortedAuthors = useMemo(() => {
    let filtered = authorsList;

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase().trim();
      filtered = authorsList.filter(author => {
        const nameMatch = author.name.toLowerCase().includes(query);
        const institutionMatch = author.institution?.toLowerCase().includes(query) || false;
        const expertiseMatch = author.expertise.some(exp => 
          exp.toLowerCase().includes(query)
        );
        return nameMatch || institutionMatch || expertiseMatch;
      });
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue: any;
      let bValue: any;

      switch (sortBy) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'institution':
          aValue = (a.institution || '').toLowerCase();
          bValue = (b.institution || '').toLowerCase();
          break;
        case 'recentActivity':
          aValue = a.recentActivity ? new Date(a.recentActivity).getTime() : 0;
          bValue = b.recentActivity ? new Date(b.recentActivity).getTime() : 0;
          break;
        case 'papers':
          aValue = a.paperCount;
          bValue = b.paperCount;
          break;
        case 'citations':
          aValue = a.totalCitations || 0;
          bValue = b.totalCitations || 0;
          break;
        case 'hIndex':
          aValue = a.hIndex || 0;
          bValue = b.hIndex || 0;
          break;
        default:
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
      }

      if (aValue < bValue) {
        return sortDirection === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortDirection === 'asc' ? 1 : -1;
      }
      return 0;
    });

    return filtered;
  }, [authorsList, searchQuery, sortBy, sortDirection]);

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const handleAuthorClick = async (authorId: string) => {
    try {
      const author = await loadAuthorClient(authorId);
      if (author) {
        setSelectedAuthor(author);
        setIsOverlayOpen(true);
      }
    } catch (error) {
      console.error('Error loading author:', error);
    }
  };

  const closeOverlay = () => {
    setIsOverlayOpen(false);
    setSelectedAuthor(null);
  };

  // ESC key closes overlay
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOverlayOpen) {
        closeOverlay();
      }
    };
    
    if (isOverlayOpen) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOverlayOpen]);

  const handleSortClick = (key: string) => {
    if (sortBy === key) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(key);
      setSortDirection('asc');
    }
  };

  const handleToggleFollow = (authorId: string) => {
    setAuthorsList(prevAuthors => 
      prevAuthors.map(author => 
        author.id === authorId 
          ? { ...author, isFollowing: !author.isFollowing }
          : author
      )
    );
  };

  const sortOptions = [
    { key: 'name', label: 'Name' },
    { key: 'institution', label: 'Institution' },
    { key: 'recentActivity', label: 'Recent Activity' },
    { key: 'papers', label: 'Papers' },
    { key: 'citations', label: 'Citations' },
    { key: 'hIndex', label: 'H-index' }
  ];

  return (
    <div className="p-4">
      {/* Filter and Sort Controls */}
      <div className="w-full mb-4 space-y-4">
        {/* Search Input Section */}
        <div className="relative">
          {/* Search Icon */}
          <svg 
            className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="8" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m21 21-4.35-4.35" />
          </svg>
          
          {/* Search Input Field */}
          <input
            ref={filterInputRef}
            type="text"
            placeholder="Filter..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            aria-label="Search and filter authors"
          />
          
          {/* Clear Button - only show when there's content */}
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Clear search"
              type="button"
            >
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
        
        {/* Sort Controls Section */}
        <div className="flex items-center gap-4 text-sm">
          {/* Sort Label */}
          <span className="text-gray-600 font-medium">Sort by:</span>
          
          {/* Sort Buttons Container */}
          <div className="flex gap-2">
            {sortOptions.map(({ key, label }) => {
              const isActive = sortBy === key;
              return (
                <button
                  key={key}
                  onClick={() => handleSortClick(key)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-primary-100 text-primary-700 border border-primary-200'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-200'
                  }`}
                >
                  {label}
                  {isActive && (
                    <span className="ml-1">
                      {sortDirection === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>
      
      {/* Authors Grid */}
      <div className="w-full px-2">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredAndSortedAuthors.map((author) => (
            <div
              key={author.id}
              className="relative break-inside-avoid mb-6"
            >
              <button
                onClick={() => handleAuthorClick(author.id)}
                className="relative w-full bg-white rounded-lg border border-gray-200 p-3 hover:shadow-md transition-all cursor-pointer min-h-[11rem] h-44 flex flex-col"
              >
                {/* Avatar and follow button section */}
                <div className="flex items-start gap-3 min-w-0 mb-3 min-h-[3.5rem] h-16 flex-shrink-0">
                  {/* Left: Avatar and Follow Button */}
                  <div className="flex flex-col items-center flex-shrink-0" style={{ width: 60 }}>
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center text-sm font-semibold ${author.claimed ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-600'}`}>
                      {author.avatar || getInitials(author.name)}
                    </div>
                    {author.claimed && (
                      <button
                        onClick={e => { 
                          e.stopPropagation(); 
                          handleToggleFollow(author.id);
                        }}
                        className={`mt-1 px-2 py-0.5 rounded-full text-[9px] uppercase tracking-wider font-semibold transition-colors ${
                          author.isFollowing
                            ? 'bg-orange-100 text-orange-700'
                            : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                        }`}
                      >
                        {author.isFollowing ? 'FOLLOWING' : 'FOLLOW'}
                      </button>
                    )}
                  </div>
                  
                  {/* Right: Name and Institution */}
                  <div className="flex-1 min-w-0">
                    <h3 className="text-base font-semibold text-gray-900 break-words leading-tight mb-1">
                      {author.name}
                    </h3>
                    <p className="text-gray-600 text-sm break-words leading-tight">
                      {author.institution || ''}
                    </p>
                  </div>
                </div>
                
                {/* Middle Section: Expertise Tags */}
                <div className="w-full">
                  <div className="text-xs font-semibold text-gray-700 leading-tight break-words">
                    {author.expertise.map((exp: string, index: number) => (
                      <React.Fragment key={exp}>
                        {index > 0 && <span className="text-gray-400"> • </span>}
                        <span 
                          className="hover:text-blue-600 transition-colors cursor-pointer"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSearchQuery(exp);
                            filterInputRef.current?.focus();
                          }}
                        >
                          {exp}
                        </span>
                      </React.Fragment>
                    ))}
                  </div>
                </div>
                
                {/* Bottom Section: Metrics */}
                <div className="flex items-center justify-evenly mt-auto pt-3 border-t border-gray-100 text-xs text-gray-600 w-full">
                  <div><span className="font-semibold text-gray-900">{author.hIndex || 0}</span> h-index</div>
                  <div><span className="font-semibold text-gray-900">{author.paperCount}</span> papers</div>
                  <div><span className="font-semibold text-gray-900">{(author.totalCitations || 0).toLocaleString()}</span> citations</div>
                </div>
              </button>
            </div>
          ))}
        </div>
        
        {/* Empty State */}
        {filteredAndSortedAuthors.length === 0 && (
          <div className="text-center py-8">
            <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <p className="text-gray-500">
              {searchQuery ? 'No authors found matching your filter.' : 'No authors found.'}
            </p>
          </div>
        )}
      </div>

      {/* Author Profile Overlay */}
      {isOverlayOpen && selectedAuthor && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center pt-12 pb-12 fadeIn"
          style={{ backgroundColor: 'rgba(0, 0, 0, 0.4)' }}
          onClick={closeOverlay}
        >
          <div
            className="bg-white rounded-lg shadow-xl max-w-2xl max-h-[80vh] overflow-y-auto"
            style={{ padding: '2rem 0.5rem', margin: 'auto 0' }}
            onClick={e => e.stopPropagation()}
            tabIndex={-1}
          >
            <AuthorProfile author={selectedAuthor} onClose={closeOverlay} />
          </div>
        </div>
      )}
    </div>
  );
}; 