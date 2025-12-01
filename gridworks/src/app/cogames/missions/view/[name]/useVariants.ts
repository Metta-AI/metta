import { parseAsArrayOf, parseAsString, useQueryState } from "nuqs";
import { useMemo } from "react";

export function useVariants() {
  const [variants, setVariants] = useQueryState(
    "variants",
    parseAsArrayOf(parseAsString, ",")
      .withDefault([])
      .withOptions({ shallow: false })
  );

  const joinedVariants = variants.join(",");
  const stableVariants = useMemo(
    () => joinedVariants.split(",").filter((v) => v),
    [joinedVariants]
  );

  return [stableVariants, setVariants] as const;
}
