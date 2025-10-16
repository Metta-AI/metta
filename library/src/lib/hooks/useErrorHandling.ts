import { useCallback, useState } from "react";

import { extractActionErrorMessage } from "@/lib/utils/errors";

type ErrorMessage = string | null;

type UseErrorHandlingOptions = {
  fallbackMessage: string;
};

export function useErrorHandling(options: UseErrorHandlingOptions) {
  const { fallbackMessage } = options;
  const [error, setError] = useState<ErrorMessage>(null);

  const handleError = useCallback(
    (possibleError: unknown) => {
      const message = extractActionErrorMessage(possibleError, fallbackMessage);
      setError(message);
    },
    [fallbackMessage]
  );

  const clearError = useCallback(() => setError(null), []);

  return {
    error,
    setError: handleError,
    clearError,
  } as const;
}
