import { useCallback, useMemo, useState } from "react";

export interface FilterSortOptions<TItem> {
  getSearchableValues?: (item: TItem) => string[];
  sorters?: Record<string, (item: TItem) => number | string | Date | null>;
  initialSearch?: string;
  initialSortKey?: string;
  initialSortDirection?: "asc" | "desc";
}

export interface UseFilterSortResult<TItem> {
  searchQuery: string;
  setSearchQuery: (value: string) => void;
  sortBy: string;
  setSortBy: (key: string) => void;
  sortDirection: "asc" | "desc";
  setSortDirection: (direction: "asc" | "desc") => void;
  filteredAndSortedItems: TItem[];
}

export function useFilterSort<TItem>(
  items: TItem[],
  options: FilterSortOptions<TItem> = {}
): UseFilterSortResult<TItem> {
  const {
    getSearchableValues,
    sorters = {},
    initialSearch = "",
    initialSortKey = Object.keys(sorters)[0] ?? "",
    initialSortDirection = "asc",
  } = options;

  const [searchQuery, setSearchQuery] = useState(initialSearch);
  const [sortBy, setSortKey] = useState(initialSortKey);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">(
    initialSortDirection
  );

  const filteredItems = useMemo(() => {
    if (!searchQuery.trim()) {
      return items;
    }

    const query = searchQuery.toLowerCase();

    return items.filter((item) => {
      if (!getSearchableValues) return false;
      return getSearchableValues(item).some((value) =>
        value.toLowerCase().includes(query)
      );
    });
  }, [items, searchQuery, getSearchableValues]);

  const filteredAndSortedItems = useMemo(() => {
    if (!sortBy || !(sortBy in sorters)) {
      return filteredItems;
    }

    const sorter = sorters[sortBy];

    return [...filteredItems].sort((a, b) => {
      const valueA = sorter(a);
      const valueB = sorter(b);

      if (valueA === valueB) return 0;

      if (valueA == null) return sortDirection === "asc" ? -1 : 1;
      if (valueB == null) return sortDirection === "asc" ? 1 : -1;

      if (typeof valueA === "string" && typeof valueB === "string") {
        return sortDirection === "asc"
          ? valueA.localeCompare(valueB)
          : valueB.localeCompare(valueA);
      }

      const numA = valueA instanceof Date ? valueA.getTime() : Number(valueA);
      const numB = valueB instanceof Date ? valueB.getTime() : Number(valueB);

      if (Number.isNaN(numA) || Number.isNaN(numB)) {
        return 0;
      }

      return sortDirection === "asc" ? numA - numB : numB - numA;
    });
  }, [filteredItems, sortBy, sorters, sortDirection]);

  const setSortBy = useCallback(
    (key: string) => {
      if (sortBy === key) {
        setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortKey(key);
      }
    },
    [sortBy]
  );

  return {
    searchQuery,
    setSearchQuery,
    sortBy,
    setSortBy,
    sortDirection,
    setSortDirection,
    filteredAndSortedItems,
  };
}
