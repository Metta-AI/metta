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

export default async function PapersPage() {
  // Load papers data from the database with current user context
  const { papers, users, interactions } = await loadPapersWithUserContext();

  return (
    <OverlayStackProvider>
      <PapersView papers={papers} users={users} interactions={interactions} />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
