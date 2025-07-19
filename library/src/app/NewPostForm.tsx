"use client";
import { useAction } from "next-safe-action/hooks";
import { FC, useState } from "react";

import { createPostAction } from "@/posts/actions/createPostAction";

/**
 * NewPostForm Component
 * 
 * Allows users to create new posts with rich content including:
 * - Post content with LaTeX support via MathJax
 * - Automatic paper detection from URLs
 * - Simple, clean interface matching the mockup
 */
export const NewPostForm: FC = () => {
  const [content, setContent] = useState('');

  const { execute } = useAction(createPostAction, {
    onSuccess: () => {
      // Reset form
      setContent('');
      // The feed is paginated, and paginated state is stored on the
      // client side only. So refreshing the entire page is the easiest way to
      // update the list of posts.
      window.location.reload();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!content.trim()) return;
    
    const formData = new FormData();
    formData.append('title', 'New Post'); // Default title
    formData.append('content', content);
    
    execute(formData);
  };

  return (
    <div className="bg-white border-b border-gray-200 p-6">
      <div className="flex gap-3">
        <textarea
          className="flex-1 min-h-[96px] max-h-32 border border-gray-200 rounded-lg px-4 py-3 resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 placeholder-gray-400 text-sm leading-relaxed"
          placeholder={`Poast away....\nInclude arXiv URLs to automatically import papers\nLaTeX supported: $x^2 + y^2 = z^2$ for inline, $$\\alpha + \\beta = \\gamma$$ for display`}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={(e) => {
            // Submit on Enter (without Shift for new line)
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        
        {/* Post button */}
        <button
          className={`px-6 py-2 rounded-lg font-medium text-sm transition-colors self-end ${
            content.trim()
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-200 text-gray-500 cursor-not-allowed'
          }`}
          disabled={!content.trim()}
          onClick={handleSubmit}
        >
          Post
        </button>
      </div>
    </div>
  );
};
