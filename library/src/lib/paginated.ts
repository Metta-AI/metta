import "server-only";

export type Paginated<T> = {
  items: T[];
  loadMore?: (limit: number) => Promise<Paginated<T>>;
};

export function findPaginated<T>(cursor: string | undefined, limit: number) {
  // TODO - implement this.
  // (I had a version of this function for Prisma, but now we use Drizzle so you'll have to rewrite it.)
}

export function makePaginated<T>(
  items: T[],
  limit: number,
  loadMore: (limit: number) => Promise<Paginated<T>>
): Paginated<T> {
  return {
    items: items.slice(0, limit),
    loadMore: items.length > limit ? loadMore : undefined,
  };
}
