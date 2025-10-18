import { loadPapersWithUserContext } from "@/posts/data/papers";
import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";
import { MeView } from "@/components/MeView";
import {
  OverlayStackProvider,
  OverlayStackRenderer,
} from "@/components/OverlayStack";

/**
 * Me Page
 *
 * This page displays the current user's profile and their starred papers.
 * Requires authentication - redirects to sign in if not authenticated.
 */

export default async function MePage() {
  // Get current user session
  const session = await auth();

  if (!session?.user?.id) {
    redirect("/api/auth/signin");
  }

  // Load papers data with current user context
  const { papers, users, interactions } = await loadPapersWithUserContext();

  // Filter papers to only show starred papers for current user
  const starredPapers = papers.filter((paper) => paper.isStarredByCurrentUser);

  return (
    <OverlayStackProvider>
      <MeView
        user={{
          id: session.user.id,
          name: session.user.name ?? null,
          email: session.user.email ?? null,
          image: session.user.image ?? null,
        }}
        starredPapers={starredPapers}
        allPapers={papers}
        users={users}
        interactions={interactions}
      />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
