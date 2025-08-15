import { loadPapersWithUserContext } from "@/posts/data/papers";
import { PapersView } from "@/components/PapersView";
import {
  OverlayStackProvider,
  OverlayStackRenderer,
} from "@/components/OverlayStack";

/**
 * Papers Page
 *
 * This page displays all papers from the database in a table format
 * with sorting, filtering, and search capabilities.
 * Includes current user context for star/queue interactions.
 */

export default async function PapersPage({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  // Load papers data from the database with current user context
  const { papers, users, interactions } = await loadPapersWithUserContext();

  // Get search parameter for initial filter
  const params = await searchParams;
  const initialSearch = typeof params.search === "string" ? params.search : "";

  return (
    <OverlayStackProvider>
      <PapersView
        papers={papers}
        users={users}
        interactions={interactions}
        initialSearch={initialSearch}
      />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
