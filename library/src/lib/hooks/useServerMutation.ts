/**
 * Standard mutation hook factory for server actions
 *
 * This creates a consistent pattern for all mutations that wrap server actions.
 * It handles FormData conversion, optimistic updates, and cache invalidation.
 *
 * @example
 * ```ts
 * import { createServerMutation } from '@/lib/hooks/useServerMutation';
 * import { toggleStarAction } from '@/posts/actions/toggleStarAction';
 *
 * export const useToggleStar = createServerMutation({
 *   mutationFn: toggleStarAction,
 *   invalidateQueries: [['papers'], ['feed']],
 * });
 * ```
 */

import {
  useMutation,
  useQueryClient,
  UseMutationOptions,
  UseMutationResult,
} from "@tanstack/react-query";

// next-safe-action returns this structure
type SafeActionResult<TData = unknown> = {
  data?: TData;
  serverError?: string;
  error?: { serverError?: string };
};

type ServerActionResult<TData = unknown> =
  | { data: TData; serverError?: never }
  | { data?: never; serverError: string };

type QueryKey = readonly unknown[];

interface ServerMutationConfig<TData, TVariables> {
  /**
   * The server action to call (returns result from next-safe-action)
   */
  mutationFn: (formData: FormData) => Promise<SafeActionResult<TData>>;

  /**
   * Convert variables to FormData
   * If not provided, assumes TVariables is an object with string values
   */
  toFormData?: (variables: TVariables) => FormData;

  /**
   * Query keys to invalidate on success
   * Can be array of query keys or a function that receives the result
   */
  invalidateQueries?:
    | QueryKey[]
    | ((data: TData, variables: TVariables) => QueryKey[]);

  /**
   * Additional mutation options
   */
  mutationOptions?: Omit<
    UseMutationOptions<TData, Error, TVariables>,
    "mutationFn" | "onSuccess"
  >;
}

/**
 * Default FormData converter for simple object types
 */
function defaultToFormData(variables: Record<string, unknown>): FormData {
  const formData = new FormData();

  for (const [key, value] of Object.entries(variables)) {
    if (value === undefined || value === null) continue;

    if (Array.isArray(value)) {
      // For arrays, append each item or stringify
      value.forEach((item) => {
        formData.append(
          key,
          typeof item === "string" ? item : JSON.stringify(item)
        );
      });
    } else if (typeof value === "object") {
      formData.append(key, JSON.stringify(value));
    } else {
      formData.append(key, String(value));
    }
  }

  return formData;
}

/**
 * Create a mutation hook that wraps a server action
 */
export function createServerMutation<
  TData = unknown,
  TVariables = Record<string, unknown>,
>(config: ServerMutationConfig<TData, TVariables>) {
  return function useServerMutation(
    options?: Omit<UseMutationOptions<TData, Error, TVariables>, "mutationFn">
  ): UseMutationResult<TData, Error, TVariables> {
    const queryClient = useQueryClient();

    const toFormData =
      config.toFormData ?? (defaultToFormData as (v: TVariables) => FormData);

    return useMutation({
      mutationFn: async (variables: TVariables) => {
        const formData = toFormData(variables);
        const result = await config.mutationFn(formData);

        // Handle next-safe-action error structure
        const errorMessage = result.serverError || result.error?.serverError;
        if (errorMessage) {
          throw new Error(errorMessage);
        }

        if (!result.data) {
          throw new Error("No data returned from server action");
        }

        return result.data;
      },
      onSuccess: (data, variables, context) => {
        // Handle query invalidation
        if (config.invalidateQueries) {
          const queryKeys =
            typeof config.invalidateQueries === "function"
              ? config.invalidateQueries(data, variables)
              : config.invalidateQueries;

          queryKeys.forEach((queryKey) => {
            queryClient.invalidateQueries({ queryKey });
          });
        }

        // Call user-provided onSuccess
        options?.onSuccess?.(data, variables, context);
      },
      ...config.mutationOptions,
      ...options,
    });
  };
}

/**
 * Standard query key factories
 * These ensure consistent query keys across the app
 */
export const queryKeys = {
  papers: {
    all: ["papers"] as const,
    lists: () => ["papers", "list"] as const,
    list: (filters: unknown) => ["papers", "list", filters] as const,
    details: () => ["papers", "detail"] as const,
    detail: (id: string) => ["papers", "detail", id] as const,
    stars: (paperId: string) => ["paper-stars", paperId] as const,
  },
  posts: {
    all: ["posts"] as const,
    lists: () => ["posts", "list"] as const,
    list: (filters: unknown) => ["posts", "list", filters] as const,
    details: () => ["posts", "detail"] as const,
    detail: (id: string) => ["posts", "detail", id] as const,
  },
  feed: {
    all: ["feed"] as const,
    list: (cursor?: unknown) => ["feed", "list", cursor] as const,
  },
  comments: {
    all: ["comments"] as const,
    byPost: (postId: string) => ["comments", postId] as const,
  },
  authors: {
    all: ["authors"] as const,
    lists: () => ["authors", "list"] as const,
    list: (filters: unknown) => ["authors", "list", filters] as const,
    details: () => ["authors", "detail"] as const,
    detail: (id: string) => ["authors", "detail", id] as const,
  },
  institutions: {
    all: ["institutions"] as const,
    lists: () => ["institutions", "list"] as const,
    list: (filters: unknown) => ["institutions", "list", filters] as const,
    details: () => ["institutions", "detail"] as const,
    detail: (id: string) => ["institutions", "detail", id] as const,
  },
  notifications: {
    all: ["notifications"] as const,
    count: ["notifications", "count"] as const,
  },
  user: {
    papers: ["user-papers"] as const,
    settings: ["user", "settings"] as const,
  },
} as const;
