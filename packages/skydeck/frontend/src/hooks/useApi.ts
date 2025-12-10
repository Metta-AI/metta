import { useCallback } from 'react';

export function useApi() {
  const apiCall = useCallback(async <T>(endpoint: string, options: RequestInit = {}): Promise<T> => {
    const response = await fetch(`/api${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'API call failed' }));
      throw new Error(error.detail || 'API call failed');
    }

    return response.json();
  }, []);

  return { apiCall };
}
