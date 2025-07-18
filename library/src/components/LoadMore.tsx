import { FC } from "react";

import { Button, type ButtonProps } from "@/components/ui/Button";

type Props = {
  loadNext: (count: number) => void;
  size?: ButtonProps["size"];
};

// Note: it's possible to refactor this component into a fully automated
// "infinite scroll" version, that would call `loadNext` when scrolled into view.
export const LoadMore: FC<Props> = ({ loadNext, size }) => {
  return (
    <div className="mt-4">
      <Button size={size} onClick={() => loadNext(20)}>
        Load more
      </Button>
    </div>
  );
};
