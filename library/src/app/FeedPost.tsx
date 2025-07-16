import Link from "next/link";
import { FC } from "react";

import { postRoute } from "@/lib/routes";
import { FeedPostDTO } from "@/posts/data/feed";

export const FeedPost: FC<{ post: FeedPostDTO }> = ({ post }) => {
  return (
    <div className="rounded border border-gray-200 p-2">
      <Link href={postRoute(post.id)} className="text-blue-600 hover:underline">
        {post.title}
      </Link>
      <div>{post.createdAt.toISOString()}</div>
    </div>
  );
};
