"use client";

import React, { ReactNode, useEffect } from "react";
import dynamic from "next/dynamic";

import {
  useOverlayStore,
  type OverlayStackItem,
  type PaperOverlayData,
  type AuthorOverlayData,
  type InstitutionOverlayData,
} from "@/lib/stores/overlayStore";
import { BaseOverlay } from "@/components/overlays/BaseOverlay";
import { OverlayLoadingSkeleton } from "@/components/overlays/OverlayLoadingSkeleton";

const NavigablePaperOverlay = dynamic(() => import("./NavigablePaperOverlay"));
const NavigableAuthorOverlay = dynamic(
  () => import("./NavigableAuthorOverlay")
);
const InstitutionOverlay = dynamic(() => import("./InstitutionOverlay"));

export const OverlayStackProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  return <>{children}</>;
};

export const OverlayStackRenderer: React.FC = () => {
  const overlays = useOverlayStore((state) => state.overlays);
  const popOverlay = useOverlayStore((state) => state.popOverlay);
  const clearStack = useOverlayStore((state) => state.clearStack);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && overlays.length > 0) {
        if (overlays.length === 1) {
          clearStack();
        } else {
          popOverlay();
        }
      }
    };

    if (overlays.length > 0) {
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }
  }, [overlays.length, clearStack, popOverlay]);

  if (overlays.length === 0) {
    return null;
  }

  const topOverlay = overlays[overlays.length - 1];

  return (
    <BaseOverlay
      open
      title={topOverlay.title}
      onClose={popOverlay}
      dismissible
      contentClassName="max-h-[70vh]"
    >
      <OverlayContent overlay={topOverlay} onClose={popOverlay} />
    </BaseOverlay>
  );
};

const OverlayContent: React.FC<{
  overlay: OverlayStackItem;
  onClose: () => void;
}> = ({ overlay, onClose }) => {
  const renderContent = () => {
    switch (overlay.type) {
      case "paper":
        const paperPayload = overlay.payload as PaperOverlayData;
        return (
          <React.Suspense fallback={<OverlayLoadingSkeleton />}>
            <NavigablePaperOverlay
              paper={paperPayload.paper}
              users={paperPayload.users}
              interactions={paperPayload.interactions}
              onClose={onClose}
              onStarToggle={paperPayload.onStarToggle}
              onQueueToggle={paperPayload.onQueueToggle}
            />
          </React.Suspense>
        );
      case "author":
        const authorPayload = overlay.payload as AuthorOverlayData;
        return (
          <React.Suspense fallback={<OverlayLoadingSkeleton />}>
            <NavigableAuthorOverlay
              author={authorPayload.author}
              onClose={onClose}
            />
          </React.Suspense>
        );
      case "institution":
        const institutionPayload = overlay.payload as InstitutionOverlayData;
        return (
          <React.Suspense fallback={<OverlayLoadingSkeleton />}>
            <InstitutionOverlay
              institution={{
                name: institutionPayload.name,
                papers: institutionPayload.papers,
                authors: institutionPayload.authors,
              }}
              onClose={onClose}
            />
          </React.Suspense>
        );
      default:
        return <div>Unknown overlay type</div>;
    }
  };

  return <div className="max-h-full overflow-hidden">{renderContent()}</div>;
};

export const useOverlayNavigation = () => {
  const pushOverlay = useOverlayStore((state) => state.pushOverlay);

  const openPaper = (
    paper: OverlayStackItem<"paper">["payload"]["paper"],
    users: OverlayStackItem<"paper">["payload"]["users"],
    interactions: OverlayStackItem<"paper">["payload"]["interactions"],
    onStarToggle: OverlayStackItem<"paper">["payload"]["onStarToggle"],
    onQueueToggle: OverlayStackItem<"paper">["payload"]["onQueueToggle"]
  ) => {
    pushOverlay(
      "paper",
      { paper, users, interactions, onStarToggle, onQueueToggle },
      paper.title
    );
  };

  const openAuthor = (
    author: OverlayStackItem<"author">["payload"]["author"]
  ) => {
    pushOverlay("author", { author }, author.name);
  };

  const openInstitution = (
    name: string,
    papers: OverlayStackItem<"institution">["payload"]["papers"],
    authors: OverlayStackItem<"institution">["payload"]["authors"]
  ) => {
    pushOverlay("institution", { name, papers, authors }, name);
  };

  return {
    openPaper,
    openAuthor,
    openInstitution,
  };
};
