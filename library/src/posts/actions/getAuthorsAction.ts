"use server";

import {
  loadAuthors,
  loadAuthor,
  type AuthorDTO,
} from "../data/authors-server";

/**
 * Server action to fetch all authors or search by name
 */
export async function getAuthorsAction(params?: {
  search?: string;
}): Promise<AuthorDTO[]> {
  const authors = await loadAuthors();

  if (params?.search) {
    const searchLower = params.search.toLowerCase();

    // Prioritize exact matches
    const exactMatches = authors.filter(
      (author) => author.name.toLowerCase() === searchLower
    );
    if (exactMatches.length > 0) {
      return exactMatches;
    }

    // Then partial matches
    return authors.filter((author) =>
      author.name.toLowerCase().includes(searchLower)
    );
  }

  return authors;
}

/**
 * Server action to fetch a single author by ID
 */
export async function getAuthorAction(
  authorId: string
): Promise<AuthorDTO | null> {
  return loadAuthor(authorId);
}
