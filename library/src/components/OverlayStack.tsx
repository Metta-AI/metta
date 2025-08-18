"use client";

import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useEffect,
} from "react";
import {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";
import { AuthorDTO } from "@/posts/data/authors-client";

/**
 * Types of overlays that can be stacked
 */
export type OverlayType = "paper" | "author" | "institution";

/**
 * Data for each overlay type
 */
export interface OverlayData {
  paper: {
    paper: PaperWithUserContext;
    users: User[];
    interactions: UserInteraction[];
    onStarToggle: (paperId: string) => void;
    onQueueToggle: (paperId: string) => void;
  };
  author: {
    author: AuthorDTO;
  };
  institution: {
    institution: {
      name: string;
      papers: PaperWithUserContext[];
      authors: AuthorDTO[];
    };
  };
}

/**
 * Individual overlay in the stack
 */
export interface StackedOverlay {
  id: string;
  type: OverlayType;
  data: OverlayData[OverlayType];
  title: string;
}

/**
 * Context for managing overlay stack
 */
interface OverlayStackContextType {
  overlays: StackedOverlay[];
  pushOverlay: <T extends OverlayType>(
    type: T,
    data: OverlayData[T],
    title: string
  ) => void;
  popOverlay: () => void;
  clearStack: () => void;
  stackDepth: number;
}

const OverlayStackContext = createContext<OverlayStackContextType | null>(null);

/**
 * Hook to use overlay stack
 */
export const useOverlayStack = () => {
  const context = useContext(OverlayStackContext);
  if (!context) {
    throw new Error(
      "useOverlayStack must be used within an OverlayStackProvider"
    );
  }
  return context;
};

/**
 * Provider component for overlay stack
 */
export const OverlayStackProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [overlays, setOverlays] = useState<StackedOverlay[]>([]);

  const pushOverlay = <T extends OverlayType>(
    type: T,
    data: OverlayData[T],
    title: string
  ) => {
    const id = `${type}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setOverlays((prev) => [...prev, { id, type, data, title }]);
  };

  const popOverlay = () => {
    setOverlays((prev) => prev.slice(0, -1));
  };

  const clearStack = () => {
    setOverlays([]);
  };

  return (
    <OverlayStackContext.Provider
      value={{
        overlays,
        pushOverlay,
        popOverlay,
        clearStack,
        stackDepth: overlays.length,
      }}
    >
      {children}
    </OverlayStackContext.Provider>
  );
};

/**
 * Component that renders the stacked overlays
 */
export const OverlayStackRenderer: React.FC = () => {
  const { overlays, popOverlay, clearStack } = useOverlayStack();

  // Handle ESC key to close all overlays
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && overlays.length > 0) {
        clearStack();
      }
    };

    if (overlays.length > 0) {
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }
  }, [overlays.length, clearStack]);

  if (overlays.length === 0) return null;

  return (
    <>
      {overlays.map((overlay, index) => {
        const isTop = index === overlays.length - 1;
        const zIndex = 50 + index;
        const opacity = 1 - (overlays.length - index - 1) * 0.1;
        const scale = 1 - (overlays.length - index - 1) * 0.02;
        const translateX = (overlays.length - index - 1) * 10;
        const translateY = (overlays.length - index - 1) * 10;

        return (
          <div
            key={overlay.id}
            className="fixed inset-0 flex items-center justify-center p-4"
            style={{ zIndex }}
          >
            {/* Backdrop - only show for top overlay - clicking outside closes ALL overlays */}
            {isTop && (
              <div
                className="absolute inset-0 bg-black/20 backdrop-blur-sm transition-opacity duration-200"
                onClick={clearStack}
              />
            )}

            {/* Overlay content */}
            <div
              className="relative transition-all duration-200 ease-out"
              style={{
                opacity,
                transform: `scale(${scale}) translate(${translateX}px, ${translateY}px)`,
                pointerEvents: isTop ? "auto" : "none",
              }}
            >
              {/* X button in overlays closes only the current overlay (goes back one level) */}
              <OverlayContent overlay={overlay} onClose={popOverlay} />
            </div>
          </div>
        );
      })}
    </>
  );
};

/**
 * Renders the appropriate overlay content based on type
 */
const OverlayContent: React.FC<{
  overlay: StackedOverlay;
  onClose: () => void;
}> = ({ overlay, onClose }) => {
  const renderContent = () => {
    switch (overlay.type) {
      case "paper":
        const PaperOverlay = React.lazy(
          () => import("./NavigablePaperOverlay")
        );
        const paperData = overlay.data as OverlayData["paper"];
        return (
          <React.Suspense fallback={<OverlayLoadingSkeleton />}>
            <PaperOverlay
              paper={paperData.paper}
              users={paperData.users}
              interactions={paperData.interactions}
              onClose={onClose}
              onStarToggle={paperData.onStarToggle}
              onQueueToggle={paperData.onQueueToggle}
            />
          </React.Suspense>
        );

      case "author":
        const AuthorOverlay = React.lazy(
          () => import("./NavigableAuthorOverlay")
        );
        const authorData = overlay.data as OverlayData["author"];
        return (
          <React.Suspense fallback={<OverlayLoadingSkeleton />}>
            <AuthorOverlay author={authorData.author} onClose={onClose} />
          </React.Suspense>
        );

      case "institution":
        const InstitutionOverlay = React.lazy(
          () => import("./InstitutionOverlay")
        );
        const institutionData = overlay.data as OverlayData["institution"];
        return (
          <React.Suspense fallback={<OverlayLoadingSkeleton />}>
            <InstitutionOverlay
              institution={institutionData.institution}
              onClose={onClose}
            />
          </React.Suspense>
        );

      default:
        return <div>Unknown overlay type</div>;
    }
  };

  return (
    <div className="max-h-[90vh] max-w-4xl overflow-hidden rounded-lg bg-white shadow-xl">
      {renderContent()}
    </div>
  );
};

/**
 * Loading skeleton for lazy-loaded overlays
 */
const OverlayLoadingSkeleton: React.FC = () => (
  <div className="max-w-4xl rounded-lg bg-white p-6 shadow-xl">
    <div className="animate-pulse">
      <div className="mb-4 h-8 rounded bg-gray-200"></div>
      <div className="space-y-3">
        <div className="h-4 w-3/4 rounded bg-gray-200"></div>
        <div className="h-4 w-1/2 rounded bg-gray-200"></div>
        <div className="h-4 w-5/6 rounded bg-gray-200"></div>
      </div>
    </div>
  </div>
);

/**
 * Hook for navigating between overlays (convenience wrapper)
 */
export const useOverlayNavigation = () => {
  const { pushOverlay } = useOverlayStack();

  const openPaper = (
    paper: PaperWithUserContext,
    users: User[],
    interactions: UserInteraction[],
    onStarToggle: (paperId: string) => void,
    onQueueToggle: (paperId: string) => void
  ) => {
    pushOverlay(
      "paper",
      { paper, users, interactions, onStarToggle, onQueueToggle },
      paper.title
    );
  };

  const openAuthor = (author: AuthorDTO) => {
    pushOverlay("author", { author }, author.name);
  };

  const openInstitution = (
    name: string,
    papers: PaperWithUserContext[],
    authors: AuthorDTO[]
  ) => {
    pushOverlay(
      "institution",
      { institution: { name, papers, authors } },
      name
    );
  };

  return {
    openPaper,
    openAuthor,
    openInstitution,
  };
};
