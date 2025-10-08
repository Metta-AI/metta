import { NextRequest, NextResponse } from "next/server";

import { loadAuthors } from "@/posts/data/authors-server";
import { withErrorHandler } from "@/lib/api/error-handler";

/**
 * API endpoint for fetching authors with optional search
 *
 * GET /api/authors
 * GET /api/authors?search=authorName
 * Returns list of authors, optionally filtered by search term
 */
export const GET = withErrorHandler(async (request: NextRequest) => {
  const { searchParams } = new URL(request.url);
  const searchTerm = searchParams.get("search");

  const authors = await loadAuthors();

  if (searchTerm) {
    // Filter authors by name (case-insensitive)
    // Prioritize exact matches first, then partial matches
    const searchLower = searchTerm.toLowerCase();
    const exactMatches = authors.filter(
      (author) => author.name.toLowerCase() === searchLower
    );

    if (exactMatches.length > 0) {
      return NextResponse.json(exactMatches);
    }

    const partialMatches = authors.filter((author) =>
      author.name.toLowerCase().includes(searchLower)
    );
    return NextResponse.json(partialMatches);
  }

  return NextResponse.json(authors);
});
