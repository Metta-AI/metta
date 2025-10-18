import {
  loadUserInstitutions,
  loadAllInstitutions,
} from "@/posts/data/managed-institutions";
import { InstitutionsDirectory } from "@/components/InstitutionsDirectory";
import {
  OverlayStackProvider,
  OverlayStackRenderer,
} from "@/components/OverlayStack";

/**
 * Institutions Page
 *
 * Displays unified institutions with both user management and paper data
 * integrated into a single cohesive view.
 */
export default async function InstitutionsPage() {
  const [userInstitutions, allInstitutions] = await Promise.all([
    loadUserInstitutions(),
    loadAllInstitutions(),
  ]);

  return (
    <OverlayStackProvider>
      <InstitutionsDirectory
        directory={allInstitutions}
        memberships={userInstitutions}
      />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
