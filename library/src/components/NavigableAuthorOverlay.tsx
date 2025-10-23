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
    // Convert paper from AuthorDTO.recentPapers to PaperWithUserContext format
    const paperWithContext = {
      id: paper.id,
      title: paper.title,
      abstract: paper.abstract,
      tags: paper.tags || [],
      link: paper.link,
      source: null,
      externalId: null,
      stars: paper.stars,
      starred: false,
      createdAt: paper.createdAt,
      updatedAt: paper.createdAt,
      authors: paper.authors || [],
      institutions: paper.institutions || [],
      isStarredByCurrentUser: false,
      isQueuedByCurrentUser: false,
    };

    // Open paper overlay with minimal context
    openPaper(
      paperWithContext,
      [], // users
      [], // interactions
      () => {}, // onStarToggle (noop)
      () => {} // onQueueToggle (noop)
    );
  };

  return (
    <AuthorProfile
      author={author}
      onClose={onClose}
      onInstitutionClick={handleInstitutionClick}
      onPaperClick={handlePaperClick}
    />
  );
}
