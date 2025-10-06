import { create } from "zustand";

export interface FilterState {
  searchQuery: string;
  sortKey: string;
  sortDirection: "asc" | "desc";
}

interface FilterStoreState extends FilterState {
  setSearchQuery: (query: string) => void;
  setSortKey: (key: string) => void;
  setSortDirection: (direction: "asc" | "desc") => void;
  reset: () => void;
}

const initialState: FilterState = {
  searchQuery: "",
  sortKey: "",
  sortDirection: "asc",
};

export const useFilterStore = create<FilterStoreState>()((set) => ({
  ...initialState,
  setSearchQuery: (searchQuery: string) => set({ searchQuery }),
  setSortKey: (sortKey: string) => set({ sortKey }),
  setSortDirection: (sortDirection: "asc" | "desc") => set({ sortDirection }),
  reset: () => set(initialState),
}));
