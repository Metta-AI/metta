"use client";
import { type FC, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { StyledLink } from "@/components/StyledLink";
import { FilterInput } from "@/components/FilterInput";
import { NoResultsMessage } from "@/components/NoResultsMessage";
import { viewConfigRoute } from "@/lib/routes";
import { GroupedConfigMakers } from "@/lib/api";

export const ConfigsPageRenderer: FC<{
  initialCfgs: GroupedConfigMakers;
}> = ({ initialCfgs }) => {
  const searchParams = useSearchParams();
  // QoL, share links with pre-filled filter.
  const initialFilter = searchParams.get("q") || "";
  const [filter, setFilter] = useState(initialFilter);

  const filtered = useMemo(() => {
    if (!filter) return initialCfgs;

    const result: GroupedConfigMakers = {};

    for (const kind of Object.keys(initialCfgs)) {
      const cfgList = initialCfgs[kind as keyof typeof initialCfgs];

      if (!cfgList) continue;

      const filteredCfgs = cfgList.filter(
        (cfg) =>
          cfg.path.toLowerCase().includes(filter.toLowerCase()) ||
          cfg.kind.toLowerCase().includes(filter.toLowerCase())
      );

      if (filteredCfgs.length > 0) {
        result[kind] = filteredCfgs;
      }
    }

    return result;
  }, [initialCfgs, filter]);

  return (
    <>
      <div className="mb-4">
        <FilterInput
          value={filter}
          onChange={setFilter}
          placeholder="Filter configs..."
        />
      </div>

      <NoResultsMessage show={Object.keys(filtered).length === 0} />

      {Object.keys(filtered).map((kind) => {
        return (
          <div key={kind} className="mb-4">
            <h2 className="text-lg font-bold">{kind}</h2>
            <ul className="space-y-2">
              {filtered[kind as keyof typeof filtered]?.map((cfg) => {
                return (
                  <li key={cfg.path}>
                    <StyledLink href={viewConfigRoute(cfg.path)}>
                      {cfg.path}
                    </StyledLink>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </>
  );
};
