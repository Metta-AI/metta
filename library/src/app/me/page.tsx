import { loadPapersWithUserContext } from "@/posts/data/papers";
import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";
import { MeView } from "@/components/MeView";

/**
 * Me Page
 * 
 * This page displays the current user's profile and their starred/queued papers.
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

    // Filter papers to only show starred and queued papers for current user
    const starredPapers = papers.filter(paper => paper.isStarredByCurrentUser);
    const queuedPapers = papers.filter(paper => paper.isQueuedByCurrentUser);

    return (
        <MeView 
            user={{
                id: session.user.id,
                name: session.user.name ?? null,
                email: session.user.email ?? null,
                image: session.user.image ?? null,
            }}
            starredPapers={starredPapers}
            queuedPapers={queuedPapers}
            allPapers={papers}
            users={users}
            interactions={interactions}
        />
    );
} 