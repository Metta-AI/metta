'use client';

import { useState, useMemo, useRef, useCallback } from 'react';
import { PaperWithUserContext, User, UserInteraction } from '@/posts/data/papers';
import { toggleStarAction } from '@/posts/actions/toggleStarAction';
import { toggleQueueAction } from '@/posts/actions/toggleQueueAction';
import PaperOverlay from './PaperOverlay';
import UserCard from './UserCard';

/**
 * PapersView Component
 * 
 * Displays a comprehensive table of papers with sorting, filtering, and interactive features.
 * This component uses native table semantics with sticky positioning for frozen columns,
 * providing robust cross-browser compatibility and proper accessibility.
 * 
 * Features:
 * - Sortable columns (all columns are clickable to sort)
 * - Interactive star/favorite toggling
 * - Clickable tags that apply search filters
 * - User hover cards showing user details
 * - Frozen title column with horizontal scrolling for other columns
 * - All columns individually resizable with proper handles
 * - Links to related papers
 * - Hover tooltips for long titles
 */

interface PapersViewProps {
  papers: PaperWithUserContext[];
  users: User[];
  interactions: UserInteraction[];
}

/**
 * Utility function to validate if a string is a valid URL
 */
const isValidUrl = (url: string): boolean => {
  if (!url || typeof url !== 'string') {
    return false;
  }
  
  try {
    const urlObj = new URL(url);
    return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
  } catch {
    return false;
  }
};

/**
 * Column configuration for the table
 */
interface ColumnConfig {
  key: string;
  label: string;
  width: number;
  minWidth: number;
  maxWidth: number;
  sortable: boolean;
  sticky?: boolean; // For frozen column
  renderHeader: (sortIndicator: React.ReactNode) => React.ReactNode;
  renderCell: (paper: PaperWithUserContext) => React.ReactNode;
}

export function PapersView({ papers, users, interactions }: PapersViewProps) {
  // State for search and filtering
  const [searchQuery, setSearchQuery] = useState('');
  const [showOnlyStarred, setShowOnlyStarred] = useState(false);
  const [sortColumn, setSortColumn] = useState<'title' | 'tags' | 'readBy' | 'queued'>('title');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  // State for column widths (all resizable)
  const [columnWidths, setColumnWidths] = useState({
    title: 400,
    tags: 300,
    readBy: 120,
    queued: 120
  });

  // Drag state for column resizing
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);
  const [mouseX, setMouseX] = useState(0);
  const draggedColumnRef = useRef<string | null>(null);
  const tableRef = useRef<HTMLTableElement>(null);



  // State for paper overlay
  const [selectedPaper, setSelectedPaper] = useState<PaperWithUserContext | null>(null);

  // State for user card
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  // State for loading indicator during filtering/sorting operations
  const [isLoading, setIsLoading] = useState(false);

  // Create a map of users for quick lookup
  const usersMap = useMemo(() => {
    const map = new Map<string, User>();
    users.forEach(user => {
      map.set(user.id, user);
    });
    return map;
  }, [users]);

  // Get users who have read a specific paper
  const getReadersForPaper = (paperId: string): User[] => {
    return interactions
      .filter(interaction => interaction.paperId === paperId && interaction.readAt)
      .map(interaction => usersMap.get(interaction.userId))
      .filter((user): user is User => user !== undefined);
  };

  // Get users who have queued a specific paper
  const getQueuedForPaper = (paperId: string): User[] => {
    return interactions
      .filter(interaction => interaction.paperId === paperId && interaction.queued)
      .map(interaction => usersMap.get(interaction.userId))
      .filter((user): user is User => user !== undefined);
  };

  // Get the number of users who have starred a specific paper
  const getStarCountForPaper = (paperId: string): number => {
    return interactions.filter(interaction => 
      interaction.paperId === paperId && interaction.starred
    ).length;
  };

  // Handle mouse down on any column resize handle
  const handleMouseDown = useCallback((e: React.MouseEvent, columnName: string) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
    isDraggingRef.current = true;
    draggedColumnRef.current = columnName;
    dragStartX.current = e.clientX;
    dragStartWidth.current = columnWidths[columnName as keyof typeof columnWidths];
    
    // Add global mouse event listeners
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [columnWidths]);

  // Handle mouse move during column resize
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDraggingRef.current || !draggedColumnRef.current) return;
    
    const deltaX = e.clientX - dragStartX.current;
    const newWidth = Math.max(100, dragStartWidth.current + deltaX);
    
    setColumnWidths(prev => ({
      ...prev,
      [draggedColumnRef.current!]: newWidth
    }));
    
    setMouseX(e.clientX);
  }, []);

  // Handle mouse up to end column resize
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    isDraggingRef.current = false;
    draggedColumnRef.current = null;
    
    // Remove global mouse event listeners
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);



  // Handle tag click to apply search filter
  const handleTagClick = (tag: string) => {
    if (!tag || typeof tag !== 'string') {
      console.warn('Invalid tag provided to handleTagClick:', tag);
      return;
    }
    
    const trimmedTag = tag.trim();
    if (!trimmedTag) {
      console.warn('Empty tag provided to handleTagClick');
      return;
    }
    
    setSearchQuery(trimmedTag);
  };

  // Handle paper title click
  const handlePaperClick = (paper: PaperWithUserContext) => {
    if (!paper || typeof paper !== 'object' || !paper.id) {
      console.warn('Invalid paper object provided to handlePaperClick:', paper);
      return;
    }
    
    setSelectedPaper(paper);
  };

  // Handle paper overlay close
  const handlePaperOverlayClose = () => {
    setSelectedPaper(null);
  };

  // Handle user card close
  const handleUserCardClose = () => {
    setSelectedUser(null);
  };

  // Handle user avatar click
  const handleUserClick = (user: User) => {
    setSelectedUser(user);
  };

  // Handle toggle star
  const handleToggleStar = async (paperId: string) => {
    if (!paperId || typeof paperId !== 'string') {
      console.warn('Invalid paperId provided to handleToggleStar:', paperId);
      return;
    }
    
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
    if (!paperId || typeof paperId !== 'string') {
      console.warn('Invalid paperId provided to handleToggleQueue:', paperId);
      return;
    }
    
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

  // Filter and sort papers
  const filteredAndSortedPapers = useMemo(() => {
    // Set loading state for large datasets
    if (papers.length > 100) {
      setIsLoading(true);
      // Use setTimeout to allow UI to update
      setTimeout(() => setIsLoading(false), 0);
    }
    
    let filtered = papers;

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(paper => {
        // Search in title
        if (paper.title.toLowerCase().includes(query)) return true;
        
        // Search in tags
        if (paper.tags && paper.tags.some(tag => tag.toLowerCase().includes(query))) return true;
        
        // Search in user names and emails from "read by" column
        const readers = getReadersForPaper(paper.id);
        if (readers.some(user => 
          (user.name && user.name.toLowerCase().includes(query)) ||
          (user.email && user.email.toLowerCase().includes(query))
        )) return true;
        
        // Search in user names and emails from "queued" column
        const queuedUsers = getQueuedForPaper(paper.id);
        if (queuedUsers.some(user => 
          (user.name && user.name.toLowerCase().includes(query)) ||
          (user.email && user.email.toLowerCase().includes(query))
        )) return true;
        
        return false;
      });
    }

    // Filter by starred status
    if (showOnlyStarred) {
      filtered = filtered.filter(paper => paper.isStarredByCurrentUser);
    }

    // Sort papers
    return filtered.sort((a, b) => {
      let aValue: string | number;
      let bValue: string | number;
      
      switch (sortColumn) {
        case 'title':
          aValue = a.title.toLowerCase();
          bValue = b.title.toLowerCase();
          break;
        case 'tags':
          // Sort by first tag
          aValue = a.tags && a.tags.length > 0 ? a.tags[0].toLowerCase() : '';
          bValue = b.tags && b.tags.length > 0 ? b.tags[0].toLowerCase() : '';
          break;
        case 'readBy':
          // Sort by number of readers
          aValue = getReadersForPaper(a.id).length;
          bValue = getReadersForPaper(b.id).length;
          break;
        case 'queued':
          // Sort by number of queued users
          aValue = getQueuedForPaper(a.id).length;
          bValue = getQueuedForPaper(b.id).length;
          break;

        default:
          aValue = a.title.toLowerCase();
          bValue = b.title.toLowerCase();
      }
      
      // Handle string comparison
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDirection === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      
      // Handle number comparison
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
      }
      
      return 0;
    });
  }, [papers, searchQuery, showOnlyStarred, sortColumn, sortDirection]);

  // Handle column sorting
  const handleSort = (col: string) => {
    if (sortColumn === col) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(col as 'title' | 'tags' | 'readBy' | 'queued');
      setSortDirection('asc');
    }
  };

  // Render sort indicator for column headers
  const renderSortIndicator = (columnName: string) => (
    <span className="ml-1">
      {sortColumn === columnName ? (
        sortDirection === 'asc' ? '↑' : '↓'
      ) : (
        <span className="text-gray-400">↕</span>
      )}
    </span>
  );

  // Render user avatar with highlighting
  const renderUserAvatar = (user: User, bgColor: string, textColor: string, index: number = 0) => {
    const userName = user.name || 'Unknown User';
    const initials = userName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
    
    // Check if user matches search query (only when there's a query)
    const query = searchQuery.toLowerCase().trim();
    const matchesQuery = query && ((user.name && user.name.toLowerCase().includes(query)) ||
                        (user.email && user.email.toLowerCase().includes(query)));
    
    return (
      <button
        key={user.id}
        onClick={() => handleUserClick(user)}
        className={`w-6 h-6 rounded-full ${matchesQuery ? 'bg-yellow-200 text-black' : 'bg-blue-600 text-white'} text-xs font-semibold flex items-center justify-center cursor-pointer border border-white hover:scale-110 transition-transform`}
        title={`Click to view ${userName}'s profile`}
      >
        {user.image || initials}
      </button>
    );
  };

  // Highlight text that matches the search query
  const highlightText = (text: string, query: string): React.ReactNode => {
    if (!query.trim()) return text;
    
    try {
      const regex = new RegExp(`(${query})`, 'gi');
      const parts = text.split(regex);
      
      return parts.map((part, index) => 
        regex.test(part) ? (
          <mark key={index} className="bg-yellow-200 rounded">
            {part}
          </mark>
        ) : part
      );
    } catch (error) {
      console.warn('Error highlighting text:', error);
      return text;
    }
  };

  // Column configurations for the table
  const columnConfigs: ColumnConfig[] = useMemo(() => [
    {
      key: 'title',
      label: 'Title',
      width: columnWidths.title,
      minWidth: 200,
      maxWidth: 800,
      sortable: true,
      sticky: true, // Frozen column
      renderHeader: (sortIndicator) => (
        <div className="flex items-center justify-between">
          <span>Title</span>
          {sortIndicator}
        </div>
      ),
      renderCell: (paper) => {
        const starCount = getStarCountForPaper(paper.id);
        const otherStarCount = paper.isStarredByCurrentUser ? starCount - 1 : starCount;
        
        return (
          <div className="flex items-center gap-2">
            <button 
              onClick={() => handleToggleStar(paper.id)} 
              className="focus:outline-none hover:scale-110 transition-transform flex-shrink-0 relative"
              aria-label={paper.isStarredByCurrentUser ? 'Remove from favorites' : 'Add to favorites'}
            >
              {paper.isStarredByCurrentUser ? (
                // Starred by me (filled and yellow)
                <div className="relative">
                  <svg className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/>
                  </svg>
                  {otherStarCount > 0 && (
                    <span className="absolute -top-1 -right-1 text-xs font-medium text-black bg-white rounded-full w-3 h-3 flex items-center justify-center">
                      {otherStarCount + 1}
                    </span>
                  )}
                </div>
              ) : starCount > 0 ? (
                // Starred only by others (gray with number)
                <div className="relative">
                  <svg className="w-4 h-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 20 20">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"/>
                  </svg>
                  <span className="absolute -top-1 -right-1 text-xs font-medium text-black bg-white rounded-full w-3 h-3 flex items-center justify-center">
                    {starCount}
                  </span>
                </div>
              ) : (
                // Not starred by anyone (empty and gray)
                <svg className="w-4 h-4 text-gray-300 hover:text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 20 20">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"/>
                </svg>
              )}
            </button>
            <button
              className="truncate block text-left hover:text-primary-600 transition-colors"
              onClick={() => handlePaperClick(paper)}
              title={paper.title}
            >
              {highlightText(paper.title, searchQuery)}
            </button>
          </div>
        );
      }
    },
    {
      key: 'tags',
      label: 'Tags',
      width: columnWidths.tags,
      minWidth: 100,
      maxWidth: 300,
      sortable: true,
      renderHeader: (sortIndicator) => (
        <div className="flex items-center justify-between">
          <span>Tags</span>
          {sortIndicator}
        </div>
      ),
      renderCell: (paper) => {
        const maxVisibleTags = 6; // Show up to 6 tags before truncating
        const tags = paper.tags || [];
        const visibleTags = tags.slice(0, maxVisibleTags);
        const hiddenTagsCount = tags.length - maxVisibleTags;
        
        return (
          <div className="flex flex-wrap gap-1 max-h-16 overflow-hidden">
            {visibleTags.map((tag, idx) => (
              <button
                key={idx}
                onClick={() => handleTagClick(tag)}
                className="inline-block bg-gray-100 text-gray-700 text-xs rounded-full px-2 py-0.5 hover:bg-gray-200 hover:text-gray-800 transition-colors cursor-pointer flex-shrink-0"
                title={`Click to filter by "${tag}"`}
              >
                {highlightText(tag, searchQuery)}
              </button>
            ))}
            {hiddenTagsCount > 0 && (
              <span className="text-xs text-gray-500 flex-shrink-0">
                +{hiddenTagsCount} more
              </span>
            )}
          </div>
        );
      }
    },
    {
      key: 'readBy',
      label: 'Read by',
      width: columnWidths.readBy,
      minWidth: 100,
      maxWidth: 200,
      sortable: true,
      renderHeader: (sortIndicator) => (
        <div className="flex items-center justify-between">
          <span>Read by</span>
          {sortIndicator}
        </div>
      ),
      renderCell: (paper) => (
        <div className="flex gap-0.5">
          {getReadersForPaper(paper.id).length > 0 ? 
            getReadersForPaper(paper.id).map((user, index) => 
              renderUserAvatar(user, '', '', index)
            ) : null
          }
        </div>
      )
    },

    {
      key: 'queued',
      label: 'Queued',
      width: columnWidths.queued,
      minWidth: 100,
      maxWidth: 200,
      sortable: true,
      renderHeader: (sortIndicator) => (
        <div className="flex items-center justify-between">
          <span>Queued</span>
          {sortIndicator}
        </div>
      ),
      renderCell: (paper) => (
        <div className="flex gap-0.5">
          {getQueuedForPaper(paper.id).length > 0 ? 
            getQueuedForPaper(paper.id).map((user, index) => 
              renderUserAvatar(user, '', '', index)
            ) : null
          }
        </div>
      )
    },

     ], [columnWidths, sortColumn, sortDirection, searchQuery, handleTagClick, renderUserAvatar, highlightText, getStarCountForPaper]);

  return (
    <div className="p-4">
      <div className="w-full max-w-full overflow-hidden px-2">
        {/* Filter and Sort Controls */}
        <div className="mb-6 space-y-4 max-w-full">
          {/* Search Input */}
          <div className="relative max-w-full">
            {/* Search Icon - positioned absolutely in the left side of the input */}
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
              type="text"
              value={searchQuery}
              onChange={(e) => {
                const value = e.target.value;
                // Validate input before calling callback
                if (typeof value === 'string') {
                  setSearchQuery(value);
                }
              }}
              placeholder="Filter..."
              className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              maxLength={200} // Prevent extremely long search queries
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

          {/* Starred Filter */}
          <div className="flex flex-wrap gap-4 items-center">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showOnlyStarred}
                onChange={(e) => setShowOnlyStarred(e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700">
                Show only starred ({papers.filter(paper => paper.isStarredByCurrentUser).length})
              </span>
            </label>
          </div>
        </div>
        
        {/* Papers Table Container */}
        <div className="relative">
          {/* Loading indicator */}
          {isLoading && (
            <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center z-20">
              <div className="text-gray-600">Filtering papers...</div>
            </div>
          )}
          


          {/* Width indicator during drag */}
          {isDragging && draggedColumnRef.current && (
            <div 
              className="fixed z-50 bg-blue-500 text-white text-xs px-2 py-1 rounded pointer-events-none"
              style={{
                left: mouseX,
                top: 10
              }}
            >
              {columnWidths[draggedColumnRef.current as keyof typeof columnWidths]}px
            </div>
          )}

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

          {/* User Card */}
          {selectedUser && (
            <UserCard
              user={selectedUser}
              allPapers={papers}
              users={users}
              interactions={interactions}
              onClose={handleUserCardClose}
            />
          )}
          
          {/* Scroll Container with Table */}
          <div className="overflow-auto border border-gray-200 rounded-lg">
            <table 
              ref={tableRef}
              className="w-full bg-white text-sm"
              style={{ tableLayout: 'fixed' }}
            >
              {/* Column Group for Width Management */}
              <colgroup>
                {columnConfigs.map((config) => (
                  <col 
                    key={config.key}
                    style={{ width: `${config.width}px` }}
                  />
                ))}
              </colgroup>
              
              {/* Table Header */}
              <thead>
                <tr className="bg-gray-50">
                  {columnConfigs.map((config) => (
                    <th 
                      key={config.key}
                      className={`px-4 py-2 text-left relative ${
                        config.sticky 
                          ? 'sticky left-0 z-10 bg-gray-50' 
                          : ''
                      } ${
                        config.sortable 
                          ? 'cursor-pointer hover:bg-gray-100 transition-colors' 
                          : ''
                      }`}
                      style={{
                        position: config.sticky ? 'sticky' : 'static',
                        left: config.sticky ? 0 : 'auto',
                        top: 0,
                        zIndex: config.sticky ? 10 : 'auto'
                      }}
                      onClick={() => {
                        if (config.sortable && !isDragging) {
                          handleSort(config.key);
                        }
                      }}
                    >
                      <div className="flex items-center justify-between">
                        {config.renderHeader(renderSortIndicator(config.key))}
                      </div>
                      {/* Resize handle */}
                      <div
                        className="absolute right-0 top-0 bottom-0 w-2 cursor-col-resize hover:bg-blue-400 transition-colors z-20"
                        onMouseDown={(e) => handleMouseDown(e, config.key)}
                        title="Drag to resize column"
                        style={{ 
                          cursor: 'col-resize',
                          userSelect: 'none'
                        }}
                      />
                    </th>
                  ))}
                </tr>
              </thead>
              
              {/* Table Body */}
              <tbody>
                {filteredAndSortedPapers.map(paper => {
                  // Validate paper object before rendering
                  if (!paper || typeof paper !== 'object' || !paper.id) {
                    console.warn('Invalid paper object found:', paper);
                    return null;
                  }
                  
                  return (
                    <tr key={paper.id} className="border-t border-gray-100 hover:bg-gray-50">
                      {columnConfigs.map((config) => (
                        <td 
                          key={config.key}
                          className={`px-4 py-2 ${
                            config.sticky 
                              ? 'sticky left-0 z-10 bg-white' 
                              : ''
                          }`}
                          style={{
                            position: config.sticky ? 'sticky' : 'static',
                            left: config.sticky ? 0 : 'auto',
                            zIndex: config.sticky ? 10 : 'auto'
                          }}
                        >
                          {config.renderCell(paper)}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
} 