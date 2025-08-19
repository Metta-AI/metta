"use client";

import React from "react";
import { AuthorDTO } from "@/posts/data/authors-client";
import { AuthorProfile } from "./AuthorProfile";
import { useOverlayNavigation } from "./OverlayStack";

interface NavigableAuthorOverlayProps {
  author: AuthorDTO;
  onClose: () => void;
}

/**
 * NavigableAuthorOverlay
 *
 * Wraps the existing AuthorProfile component and adds navigation capabilities
 * to related institutions and papers. This allows clicking on institution names
 * or papers to open new overlays in the stack.
 */
export default function NavigableAuthorOverlay({
  author,
  onClose,
}: NavigableAuthorOverlayProps) {
  const { openInstitution, openPaper } = useOverlayNavigation();

  // Handle institution click
  const handleInstitutionClick = (institutionName: string) => {
    if (institutionName) {
      // Let the institution overlay load the full institution data
      // by not providing any initial papers/authors arrays
      openInstitution(institutionName, [], []);
    }
  };

  // Handle paper click
  const handlePaperClick = (paper: any) => {
    // For now, we'll need to provide minimal required props
    // In a real implementation, you'd fetch the full paper data with user context
    const mockUsers: any[] = [];
    const mockInteractions: any[] = [];
    const mockOnStarToggle = (paperId: string) =>
      console.log("Star toggle:", paperId);
    const mockOnQueueToggle = (paperId: string) =>
      console.log("Queue toggle:", paperId);

    // Note: This would need proper paper data with user context in a real implementation
    // openPaper(paper, mockUsers, mockInteractions, mockOnStarToggle, mockOnQueueToggle);
    console.log(
      "Paper navigation not yet fully implemented - need PaperWithUserContext"
    );
  };

  return (
    <AuthorProfile
      author={author}
      onClose={onClose}
      onInstitutionClick={handleInstitutionClick}
    />
  );
}
