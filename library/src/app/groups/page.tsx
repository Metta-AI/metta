import { loadUserGroups, loadAllDiscoverableGroups } from "@/posts/data/groups";
import { loadUserInstitutions } from "@/posts/data/managed-institutions";
import { GroupsView } from "@/components/GroupsView";
import {
  OverlayStackProvider,
  OverlayStackRenderer,
} from "@/components/OverlayStack";

/**
 * Groups Page
 *
 * Displays groups with both user membership and public group discovery
 * integrated into a single cohesive view.
 */
export default async function GroupsPage() {
  const [userGroups, allGroups, userInstitutions] = await Promise.all([
    loadUserGroups(),
    loadAllDiscoverableGroups(),
    loadUserInstitutions(),
  ]);

  return (
    <OverlayStackProvider>
      <GroupsView
        userGroups={userGroups}
        allGroups={allGroups}
        userInstitutions={userInstitutions.map((inst) => ({
          id: inst.id,
          name: inst.name,
        }))}
      />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
