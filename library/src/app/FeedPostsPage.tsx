"use client";

import { FC } from "react";

import { LoadMore } from "@/components/LoadMore";
import { usePaginator } from "@/lib/hooks/usePaginator";
import { Paginated } from "@/lib/paginated";
import { FeedPostDTO } from "@/posts/data/feed";

import { FeedPost } from "./FeedPost";
import { NewPostForm } from "./NewPostForm";

/**
 * FeedPostsPage Component
 * 
 * Displays the main social feed with:
 * - Post composition form at the top
 * - Chronological list of posts
 * - Infinite scrolling with load more functionality
 * - Rich post display with author info and social metrics
 */
export const FeedPostsPage: FC<{
  posts: Paginated<FeedPostDTO>;
}> = ({ posts: initialPosts }) => {
  const page = usePaginator(initialPosts);

  return (
    <>
      {/* Post Composition */}
      <NewPostForm />

      {/* Feed */}
      <div className="max-w-2xl mx-auto">
        {page.items.length > 0 ? (
          <>
            {page.items.map((post) => (
              <FeedPost key={post.id} post={post} />
            ))}
            {page.loadNext && (
              <div className="p-4 border-t border-gray-200 bg-white">
                <LoadMore loadNext={page.loadNext} />
              </div>
            )}
          </>
        ) : (
          <div className="bg-white border border-gray-200 rounded-lg p-8 text-center">
            <div className="text-gray-400 mb-4">
              <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No posts yet</h3>
            <p className="text-gray-600">
              Be the first to share your research insights!
            </p>
          </div>
        )}
      </div>
    </>
  );
};
