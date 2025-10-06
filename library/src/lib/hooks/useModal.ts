import { useCallback, useState } from "react";

export function useModal<TData = void>() {
  const [isOpen, setIsOpen] = useState(false);
  const [data, setData] = useState<TData | null>(null);

  const open = useCallback((payload?: TData) => {
    setData(payload ?? null);
    setIsOpen(true);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
    setData(null);
  }, []);

  return {
    isOpen,
    data,
    open,
    close,
  } as const;
}
