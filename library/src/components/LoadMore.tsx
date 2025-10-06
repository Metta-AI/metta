import { FC } from "react";

import { Button } from "@/components/ui/button";

interface LoadMoreProps {
  loadNext: (limit?: number) => void;
  isLoading?: boolean;
  label?: string;
  limit?: number;
}

export const LoadMore: FC<LoadMoreProps> = ({
  loadNext,
  isLoading = false,
  label = "Load more",
  limit = 20,
}) => {
  return (
    <div className="mt-4">
      <Button onClick={() => loadNext(limit)} disabled={isLoading}>
        {isLoading ? "Loading..." : label}
      </Button>
    </div>
  );
};
