import { loadAuthors } from "@/posts/data/authors-server";

import { AuthorsView } from "@/components/AuthorsView";

/**
 * Authors Page
 * 
 * Displays a grid of all authors in the system with their key information
 * including name, institution, expertise, and recent activity.
 */
export default async function AuthorsPage() {
  const authors = await loadAuthors();

  return <AuthorsView authors={authors} />;
} 