import {
  loadUserInstitutions,
  loadAllInstitutions,
} from "@/posts/data/managed-institutions";
import { UnifiedInstitutionsView } from "@/components/UnifiedInstitutionsView";
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
      <UnifiedInstitutionsView
        userInstitutions={userInstitutions}
        allInstitutions={allInstitutions}
      />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
