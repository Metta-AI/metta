import { loadUserGroups, loadAllPublicGroups } from "@/posts/data/groups";
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
  const [userGroups, allGroups] = await Promise.all([
    loadUserGroups(),
    loadAllPublicGroups(),
  ]);

  return (
    <OverlayStackProvider>
      <GroupsView userGroups={userGroups} allGroups={allGroups} />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
