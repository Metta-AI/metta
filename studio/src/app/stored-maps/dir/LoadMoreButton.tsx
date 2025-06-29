"use client";
import { useQueryState } from "nuqs";

import { Button } from "@/components/Button";

import { parseLimitParam } from "./params";

export const LoadMoreButton = () => {
  const [limit, setLimit] = useQueryState(
    "limit",
    parseLimitParam.withOptions({ shallow: false })
  );
  return <Button onClick={() => setLimit(limit + 20)}>Load more</Button>;
};
