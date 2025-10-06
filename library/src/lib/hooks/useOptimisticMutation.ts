import { useCallback, useState } from "react";

interface OptimisticMutationOptions<TData, TVariables> {
  onMutate?: (variables: TVariables) => TData;
  onSuccess?: (data: TData, variables: TVariables) => void;
  onError?: (
    error: unknown,
    variables: TVariables,
    rollbackValue?: TData
  ) => void;
  mutateFn: (variables: TVariables) => Promise<TData>;
}

export function useOptimisticMutation<TData, TVariables>(
  options: OptimisticMutationOptions<TData, TVariables>
) {
  const { onMutate, onSuccess, onError, mutateFn } = options;
  const [isLoading, setIsLoading] = useState(false);

  const mutate = useCallback(
    async (variables: TVariables) => {
      let rollbackValue: TData | undefined;
      try {
        setIsLoading(true);
        rollbackValue = onMutate?.(variables);
        const result = await mutateFn(variables);
        onSuccess?.(result, variables);
        return result;
      } catch (error) {
        onError?.(error, variables, rollbackValue);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    [mutateFn, onMutate, onSuccess, onError]
  );

  return {
    mutate,
    isLoading,
  } as const;
}
