import { NextRequest, NextResponse } from "next/server";

import { listPosts } from "@/posts/data/posts-server";
import { withErrorHandler } from "@/lib/api/error-handler";

export const GET = withErrorHandler(async (request: NextRequest) => {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit");
  const offset = searchParams.get("offset");

  const response = await listPosts({
    limit: limit ? Number(limit) : undefined,
    offset: offset ? Number(offset) : undefined,
  });

  return NextResponse.json(response);
});
