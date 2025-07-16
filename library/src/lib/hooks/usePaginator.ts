import { useCallback, useState } from "react";

import { Paginated } from "@/lib/paginated";

type FullPaginated<T> = {
  items: T[];
  // This is intentionally named `loadNext` instead of `loadMore` to avoid confusion.
  loadNext?: (limit: number) => void;
  // Helper functions - if the server action has affected some items, and you
  // want to update the list without reloading the page, we need to update the
  // items list.
  // This is similar to Relay's edge directives
  // (https://relay.dev/docs/guided-tour/list-data/updating-connections/), but
  // more manual.
  prepend: (item: T) => void;
  append: (item: T) => void;
  remove: (compare: (item: T) => boolean) => void;
  update: (update: (item: T) => T) => void;
};

// Turn the initial set of items that was loaded in a server component into a
// dynamic list of items and helper methods that can fetch more items.
//
// The parameter, `initialPage`, is used only on the initial render.
export function usePaginator<T>(initialPage: Paginated<T>): FullPaginated<T> {
  const [{ items, loadMore }, setPage] = useState(initialPage);

  const prepend = useCallback((newItem: T) => {
    setPage(({ items, loadMore }) => ({
      items: [newItem, ...items],
      loadMore,
    }));
  }, []);

  const append = useCallback((newItem: T) => {
    setPage(({ items, loadMore }) => ({
      items: [...items, newItem],
      loadMore,
    }));
  }, []);

  const remove = useCallback((compare: (item: T) => boolean) => {
    setPage(({ items, loadMore }) => ({
      items: items.filter((i) => !compare(i)),
      loadMore,
    }));
  }, []);

  const update = useCallback((update: (item: T) => T) => {
    setPage(({ items, loadMore }) => ({
      items: items.map(update),
      loadMore,
    }));
  }, []);

  return {
    items,
    loadNext: loadMore
      ? (limit: number) => {
          loadMore(limit).then(({ items: newItems, loadMore: newLoadMore }) => {
            // In theory, there should be no duplicates, if `loadMore` is implemented correctly.
            // But maybe we should check for duplicate keys and skip them.
            // This would require a separate (optional?) `getKey` parameter to this hook.
            setPage(({ items }) => ({
              items: [...items, ...newItems],
              loadMore: newLoadMore,
            }));
          });
        }
      : undefined,
    append,
    prepend,
    remove,
    update,
  };
}
