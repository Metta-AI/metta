"use client";

import React from "react";
import { AuthorDTO } from "@/posts/data/authors-client";
import { AuthorProfile } from "./AuthorProfile";
import { useOverlayNavigation } from "./OverlayStack";
import * as papersApi from "@/lib/api/resources/papers";

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
  const handlePaperClick = async (paper: any) => {
    // Fetch full paper data to get all authors
    // (AuthorDTO.recentPapers may not have full author data for performance)
    try {
      const fullPaper = await papersApi.getPaper(paper.id);

      const paperWithContext = {
        id: fullPaper.id,
        title: fullPaper.title,
        abstract: fullPaper.abstract,
        tags: fullPaper.tags || [],
        link: fullPaper.link,
        source: fullPaper.source,
        externalId: null,
        stars: fullPaper.stars,
        starred: false,
        createdAt: new Date(fullPaper.createdAt),
        updatedAt: new Date(fullPaper.createdAt),
        authors: fullPaper.authors || [],
        institutions: paper.institutions || [],
        isStarredByCurrentUser: false,
      };

      // Open paper overlay with minimal context
      openPaper(
        paperWithContext,
        [], // users
        [], // interactions
        () => {} // onStarToggle (noop)
      );
    } catch (error) {
      console.error("Error loading paper:", error);
    }
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
