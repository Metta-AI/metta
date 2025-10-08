import { NextRequest, NextResponse } from "next/server";

import { loadAuthor } from "@/posts/data/authors-server";
import { NotFoundError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";

/**
 * API endpoint for fetching individual author data
 *
 * GET /api/authors/[authorId]
 * Returns detailed author information including papers and statistics
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ authorId: string }> }
) {
  try {
    const { authorId } = await params;
    const author = await loadAuthor(authorId);

    if (!author) {
      throw new NotFoundError("Author", authorId);
    }

    return NextResponse.json(author);
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/authors/[authorId]" });
  }
}
