"use client";

import { useQuery } from "@tanstack/react-query";
import * as postsApi from "@/lib/api/resources/posts";

export function usePosts(query?: { limit?: number; offset?: number }) {
  return useQuery({
    queryKey: ["posts", query],
    queryFn: () => postsApi.listPosts(query),
  });
}

export function usePost(id: string) {
  return useQuery({
    queryKey: ["posts", id],
    queryFn: () => postsApi.getPost(id),
    enabled: !!id,
  });
}
