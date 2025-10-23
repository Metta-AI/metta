import { loadUserGroups, loadAllDiscoverableGroups } from "@/posts/data/groups";
import { loadUserInstitutions } from "@/posts/data/managed-institutions";
import { GroupsView } from "@/components/GroupsView";
import {
  OverlayStackProvider,
  OverlayStackRenderer,
} from "@/components/OverlayStack";
import { auth } from "@/lib/auth";

/**
 * Groups Page
 *
 * Displays groups with both user membership and public group discovery
 * integrated into a single cohesive view.
 */
export default async function GroupsPage() {
  const [userGroups, allGroups, userInstitutions, session] = await Promise.all([
    loadUserGroups(),
    loadAllDiscoverableGroups(),
    loadUserInstitutions(),
    auth(),
  ]);

  // Only pass user if they have a valid session
  const currentUser = session?.user?.id
    ? {
        id: session.user.id,
        name: session.user.name,
        email: session.user.email,
      }
    : null;

  return (
    <OverlayStackProvider>
      <GroupsView
        userGroups={userGroups}
        allGroups={allGroups}
        userInstitutions={userInstitutions.map((inst) => ({
          id: inst.id,
          name: inst.name,
        }))}
        currentUser={currentUser}
      />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
