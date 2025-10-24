import {
  getAuthorsAction,
  getAuthorAction,
} from "@/posts/actions/getAuthorsAction";
import { type AuthorDTO } from "@/posts/data/authors-server";

export type AuthorDetail = AuthorDTO;
export type AuthorSummary = AuthorDTO;

export async function listAuthors(params?: {
  search?: string;
}): Promise<AuthorSummary[]> {
  return getAuthorsAction(params);
}

export async function getAuthor(id: string): Promise<AuthorDetail | null> {
  return getAuthorAction(id);
}
