import { NextRequest, NextResponse } from "next/server";

import { loadAuthor } from "@/posts/data/authors-server";

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
      return NextResponse.json(
        { error: "Author not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(author);
  } catch (error) {
    console.error("Error loading author:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
} 