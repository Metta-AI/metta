import { create } from "zustand";

import type { AuthorDTO } from "@/posts/data/authors-client";
import type {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";

export type OverlayType = "paper" | "author" | "institution";

export type PaperOverlayData = {
  paper: PaperWithUserContext;
  users: User[];
  interactions: UserInteraction[];
  onStarToggle: (paperId: string) => void;
  onQueueToggle: (paperId: string) => void;
};

export type AuthorOverlayData = {
  author: AuthorDTO;
};

export type InstitutionOverlayData = {
  name: string;
  papers: PaperWithUserContext[];
  authors: AuthorDTO[];
};

type OverlayPayload = {
  paper: PaperOverlayData;
  author: AuthorOverlayData;
  institution: InstitutionOverlayData;
};

export interface OverlayStackItemBase<T extends OverlayType = OverlayType> {
  id: string;
  type: T;
  payload: OverlayPayload[T];
  title: string;
  depth: number;
}

export type OverlayStackItem<T extends OverlayType = OverlayType> =
  OverlayStackItemBase<T>;

export interface OverlayStoreState {
  overlays: OverlayStackItem[];
  pushOverlay: <T extends OverlayType>(
    type: T,
    payload: OverlayPayload[T],
    title: string
  ) => void;
  popOverlay: () => void;
  clearStack: () => void;
}

export const useOverlayStore = create<OverlayStoreState>()((set) => ({
  overlays: [],
  pushOverlay: (type, payload, title) => {
    const id = `${type}-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2, 11)}`;
    set((state) => ({
      overlays: [
        ...state.overlays,
        {
          id,
          type,
          payload,
          title,
          depth: state.overlays.length,
        },
      ],
    }));
  },
  popOverlay: () => set((state) => ({ overlays: state.overlays.slice(0, -1) })),
  clearStack: () => set({ overlays: [] }),
}));
