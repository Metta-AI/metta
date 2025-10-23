"use client";

import { FC } from "react";
import { UnifiedInstitutionsView } from "./UnifiedInstitutionsView";
import { UnifiedInstitutionDTO } from "@/posts/data/managed-institutions";

// Legacy compatibility - redirect to UnifiedInstitutionsView
interface ManagedInstitutionsViewProps {
  userInstitutions: UnifiedInstitutionDTO[];
  allInstitutions: UnifiedInstitutionDTO[];
}

export const ManagedInstitutionsView: FC<ManagedInstitutionsViewProps> = ({
  userInstitutions,
  allInstitutions,
}) => {
  return (
    <UnifiedInstitutionsView
      userInstitutions={userInstitutions}
      allInstitutions={allInstitutions}
    />
  );
};
