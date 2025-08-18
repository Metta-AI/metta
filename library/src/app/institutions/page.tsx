import { loadInstitutions } from "@/posts/data/institutions-server";
import { InstitutionsView } from "@/components/InstitutionsView";
import {
  OverlayStackProvider,
  OverlayStackRenderer,
} from "@/components/OverlayStack";

/**
 * Institutions Page
 *
 * Displays a grid of all institutions in the system with their key information
 * including name, paper count, author count, and research areas.
 */
export default async function InstitutionsPage() {
  const institutions = await loadInstitutions();

  return (
    <OverlayStackProvider>
      <InstitutionsView institutions={institutions} />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
