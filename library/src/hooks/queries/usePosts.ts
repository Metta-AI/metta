"use client";

import { useQuery } from "@tanstack/react-query";
import * as postsApi from "@/lib/api/resources/posts";

// Note: usePosts() removed - use server-side data fetching with posts-server.ts instead
// The listPosts API was intentionally removed

export function usePost(id: string) {
  return useQuery({
    queryKey: ["posts", id],
    queryFn: () => postsApi.getPost(id),
    enabled: !!id,
  });
}
