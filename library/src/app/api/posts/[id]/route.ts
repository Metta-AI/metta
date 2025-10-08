import { NextRequest, NextResponse } from "next/server";

import loadPost from "@/posts/data/post";
import { NotFoundError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const post = await loadPost(id);
    return NextResponse.json(post);
  } catch (error) {
    // Check if it's a "not found" error
    if (error instanceof Error && error.message.includes("not found")) {
      return handleApiError(new NotFoundError("Post"), {
        endpoint: "GET /api/posts/[id]",
      });
    }
    return handleApiError(error, { endpoint: "GET /api/posts/[id]" });
  }
}
