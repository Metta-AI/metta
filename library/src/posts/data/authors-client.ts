import { authors } from "@/lib/api/resources";

export type AuthorDTO = authors.AuthorDetail;

export async function loadAuthorClient(
  authorId: string
): Promise<AuthorDTO | null> {
  return authors.getAuthor(authorId);
}
