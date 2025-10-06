"use client";
import clsx from "clsx";
import { FC, useMemo, useState } from "react";
import z from "zod/v4";

import { ConfigViewer } from "@/components/ConfigViewer";
import { Tabs } from "@/components/Tabs";
import { Config } from "@/lib/api";

import { MapSection } from "./MapSection";

const namedMettaGridConfigsSchema = z.record(z.string(), z.unknown());

export const ExploreNamedMettaGridConfigs: FC<{ cfg: Config }> = ({ cfg }) => {
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);

  const selectedSimulationConfig = useMemo(() => {
    return selectedConfig
      ? namedMettaGridConfigsSchema.parse(cfg.config.value)[selectedConfig]
      : null;
  }, [cfg.config.value, selectedConfig]);

  const namedMettaGridConfigs = namedMettaGridConfigsSchema.parse(
    cfg.config.value
  );

  return (
    <div className="flex gap-8">
      <div>
        <h2 className="mb-2 ml-4 font-semibold">Simulations</h2>
        <div className="flex flex-col">
          {Object.keys(namedMettaGridConfigs).map((name) => (
            <div
              key={name}
              className={clsx(
                "cursor-pointer rounded-md px-4 py-1 text-sm",
                selectedConfig === name ? "bg-blue-200" : "hover:bg-blue-100"
              )}
              onClick={() => setSelectedConfig(name)}
            >
              {name}
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1">
        {selectedConfig && selectedSimulationConfig ? (
          <Tabs
            topPadding
            tabs={[
              {
                id: "map",
                label: "Map",
                content: (
                  <MapSection
                    key={selectedConfig}
                    cfg={cfg}
                    name={selectedConfig!}
                  />
                ),
              },
              {
                id: "config",
                label: "Config",
                content: (
                  <ConfigViewer
                    key={selectedConfig}
                    value={selectedSimulationConfig}
                  />
                ),
              },
            ]}
          ></Tabs>
        ) : (
          <div className="pt-4 text-center text-gray-500">
            No config selected
          </div>
        )}
      </div>
    </div>
  );
};
