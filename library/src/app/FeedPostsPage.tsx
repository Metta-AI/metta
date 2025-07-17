"use client";

import { FC } from "react";

import { LoadMore } from "@/components/LoadMore";
import { usePaginator } from "@/lib/hooks/usePaginator";
import { Paginated } from "@/lib/paginated";
import { FeedPostDTO } from "@/posts/data/feed";

import { FeedPost } from "./FeedPost";
import { NewPostForm } from "./NewPostForm";

export const FeedPostsPage: FC<{
  posts: Paginated<FeedPostDTO>;
}> = ({ posts: initialPosts }) => {
  const page = usePaginator(initialPosts);

  return (
    <div>
      <div>
        <NewPostForm />
      </div>
      <div className="mt-4">
        <div>
          <div className="flex flex-col gap-2">
            {page.items.map((post) => (
              <FeedPost key={post.id} post={post} />
            ))}
          </div>
          {page.loadNext && <LoadMore loadNext={page.loadNext} />}
        </div>
      </div>
    </div>
  );
};
