"use client";

import { FC } from "react";
import { LoadMore } from "@/components/LoadMore";
import { usePaginator } from "@/lib/hooks/usePaginator";
import { Paginated } from "@/lib/paginated";
import { FeedPostDTO } from "@/posts/data/feed";
import { FeedPost } from "@/app/FeedPost";
import { NewPostForm } from "@/app/NewPostForm";

interface LibraryContentProps {
  activeNav: string;
  initialPosts?: Paginated<FeedPostDTO>;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
}

export const LibraryContent: FC<LibraryContentProps> = ({
  activeNav,
  initialPosts,
  currentUser,
}) => {
  // For now, we'll show the feed view as default
  // In the future, this will be determined by the activeNav prop
  const page = usePaginator(initialPosts || { items: [], loadMore: undefined });

  return (
    <div className="flex-1 p-6">
      <div className="mx-auto max-w-4xl">
        <div>
          <NewPostForm />
        </div>
        <div className="mt-4">
          <div>
            <div className="flex flex-col gap-2">
              {page.items?.map((post) => (
                <FeedPost key={post.id} post={post} currentUser={currentUser} />
              ))}
            </div>
            {page.loadNext && <LoadMore loadNext={page.loadNext} />}
          </div>
        </div>
      </div>
    </div>
  );
};
