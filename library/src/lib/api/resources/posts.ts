import { getPostAction } from "@/posts/actions/getPostAction";
import { type PostDTO } from "@/posts/data/post";

export type PostDetail = PostDTO;

export interface PostSummary {
  id: string;
  title: string;
  createdAt: string;
  author: {
    id: string;
    name: string | null;
  };
  paperId: string | null;
  stars: number;
}

export interface PostListResponse {
  posts: PostSummary[];
  pagination: {
    limit: number;
    offset: number;
    total: number;
  };
}

// Note: listPosts() removed - API route was deleted
// Use server-side data fetching instead

export async function getPost(id: string): Promise<PostDetail> {
  return getPostAction(id);
}
