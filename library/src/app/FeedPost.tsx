"use client";

import Link from "next/link";
import { FC, useState } from "react";

import { postRoute } from "@/lib/routes";
import { FeedPostDTO } from "@/posts/data/feed";

/**
 * FeedPost Component
 * 
 * Displays a single post in the social feed with rich formatting including:
 * - Author information with avatar
 * - Post content with LaTeX support
 * - Social metrics (likes, retweets, replies)
 * - Paper references when applicable
 * - Interactive elements
 */
export const FeedPost: FC<{ post: FeedPostDTO }> = ({ post }) => {
  const [expandedAbstract, setExpandedAbstract] = useState(false);

  // Generate user initials for avatar
  const getUserInitials = (name: string | null, email: string | null) => {
    if (name) {
      return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
    }
    if (email) {
      return email.charAt(0).toUpperCase();
    }
    return '?';
  };

  // Format relative time
  const formatRelativeTime = (date: Date) => {
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'now';
    if (diffInHours < 24) return `${diffInHours}h`;
    
    const diffInDays = Math.floor(diffInHours / 24);
    if (diffInDays < 7) return `${diffInDays}d`;
    
    const diffInWeeks = Math.floor(diffInDays / 7);
    if (diffInWeeks < 4) return `${diffInWeeks}w`;
    
    const diffInMonths = Math.floor(diffInDays / 30);
    return `${diffInMonths}m`;
  };

  // Handle social interactions
  const handleLike = () => {
    // TODO: Implement like functionality
    console.log('Like post:', post.id);
  };

  const handleRetweet = () => {
    // TODO: Implement retweet functionality
    console.log('Retweet post:', post.id);
  };

  const handleReply = () => {
    // TODO: Implement reply functionality
    console.log('Reply to post:', post.id);
  };

  // Handle pure paper posts (papers without user commentary)
  if (post.postType === 'pure-paper' && post.paper) {
    return (
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-gray-900 mb-2 leading-tight">
                {post.paper.link ? (
                  <a 
                    href={post.paper.link} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="hover:text-blue-600 transition-colors"
                  >
                    {post.paper.title}
                  </a>
                ) : (
                  post.paper.title
                )}
              </h3>
              {post.paper.authors && post.paper.authors.length > 0 && (
                <p className="text-sm text-gray-600 mb-2">
                  {post.paper.authors.join(', ')}
                </p>
              )}
            </div>
            <div className="flex items-center gap-2 ml-4">
              <button
                onClick={() => setExpandedAbstract(!expandedAbstract)}
                className="text-xs text-blue-600 hover:text-blue-700 font-medium"
              >
                {expandedAbstract ? 'Hide Abstract' : 'Show Abstract'}
              </button>
            </div>
          </div>
          
          {expandedAbstract && post.paper.abstract && (
            <div className="border-t border-gray-100 pt-3">
              <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                {post.paper.abstract}
              </p>
            </div>
          )}
          
          {post.paper.tags && post.paper.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-3">
              {post.paper.tags.map((tag: string, index: number) => (
                <span 
                  key={index} 
                  className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Handle user posts and paper posts with commentary
  return (
    <div 
      className={`border-b border-gray-200 p-6 ${
        post.postType === 'user-post' ? 'bg-blue-50' :
        post.postType === 'paper-post' ? 'bg-orange-50' : 'bg-white'
      }`}
    >
      {/* Post header with user info */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-semibold">
          {getUserInitials(post.author.name, post.author.email)}
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="font-semibold text-gray-900">
            {post.author.name || post.author.email?.split('@')[0] || 'Unknown User'}
          </span>
          <span className="text-gray-500">·</span>
          <span className="text-gray-500">{formatRelativeTime(post.createdAt)}</span>
        </div>
      </div>

      {/* Post content */}
      {post.content && (
        <div className="text-gray-900 leading-relaxed mb-4 whitespace-pre-wrap">
          {post.content}
        </div>
      )}

      {/* Embedded paper card for paper posts */}
      {post.postType === 'paper-post' && post.paper && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-4">
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1 min-w-0">
              <h4 className="text-lg font-semibold text-gray-900 mb-2 leading-tight">
                {post.paper.link ? (
                  <a 
                    href={post.paper.link} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="hover:text-blue-600 transition-colors"
                  >
                    {post.paper.title}
                  </a>
                ) : (
                  post.paper.title
                )}
              </h4>
              {post.paper.authors && post.paper.authors.length > 0 && (
                <p className="text-sm text-gray-600 mb-2">
                  {post.paper.authors.join(', ')}
                </p>
              )}
            </div>
            <div className="flex items-center gap-2 ml-4">
              <button
                onClick={() => setExpandedAbstract(!expandedAbstract)}
                className="text-xs text-blue-600 hover:text-blue-700 font-medium"
              >
                {expandedAbstract ? 'Hide Abstract' : 'Show Abstract'}
              </button>
            </div>
          </div>
          
          {expandedAbstract && post.paper.abstract && (
            <div className="border-t border-gray-100 pt-3">
              <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                {post.paper.abstract}
              </p>
            </div>
          )}
          
          {post.paper.tags && post.paper.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-3">
              {post.paper.tags.map((tag: string, index: number) => (
                <span 
                  key={index} 
                  className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Post interaction buttons */}
      <div className="flex items-center gap-6 text-gray-500">
        {/* Reply button */}
        <button 
          onClick={handleReply}
          className="flex items-center gap-2 hover:text-gray-700 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <span className="text-sm">{post.replies}</span>
        </button>
        
        {/* Retweet button */}
        <button 
          onClick={handleRetweet}
          className="flex items-center gap-2 hover:text-gray-700 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
          </svg>
          <span className="text-sm">{post.retweets}</span>
        </button>
        
        {/* Like button */}
        <button 
          onClick={handleLike}
          className="flex items-center gap-2 hover:text-red-500 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
          <span className="text-sm">{post.likes}</span>
        </button>
        
        {/* Share button */}
        <button className="flex items-center gap-2 hover:text-gray-700 transition-colors">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
          </svg>
        </button>
      </div>
    </div>
  );
};
