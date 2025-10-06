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
  const [selectedName, setSelectedName] = useState<string | null>(null);

  const selectedConfig = useMemo(() => {
    return selectedName
      ? namedMettaGridConfigsSchema.parse(cfg.config.value)[selectedName]
      : null;
  }, [cfg.config.value, selectedName]);

  const namedMettaGridConfigs = namedMettaGridConfigsSchema.parse(
    cfg.config.value
  );

  return (
    <div className="flex gap-8">
      <div>
        <h2 className="mb-2 ml-4 font-semibold">Configs</h2>
        <div className="flex flex-col">
          {Object.keys(namedMettaGridConfigs).map((name) => (
            <div
              key={name}
              className={clsx(
                "cursor-pointer rounded-md px-4 py-1 text-sm",
                name === selectedName ? "bg-blue-200" : "hover:bg-blue-100"
              )}
              onClick={() => setSelectedName(name)}
            >
              {name}
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1">
        {selectedName && selectedConfig ? (
          <Tabs
            topPadding
            tabs={[
              {
                id: "map",
                label: "Map",
                content: (
                  <MapSection
                    key={selectedName}
                    cfg={cfg}
                    name={selectedName!}
                  />
                ),
              },
              {
                id: "config",
                label: "Config",
                content: (
                  <ConfigViewer
                    key={selectedName}
                    value={selectedConfig}
                    kind="MettaGridConfig"
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
