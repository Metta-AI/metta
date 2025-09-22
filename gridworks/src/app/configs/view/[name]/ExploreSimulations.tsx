"use client";
import clsx from "clsx";
import { FC, useMemo, useState } from "react";
import z from "zod/v4";

import { ConfigViewer } from "@/components/ConfigViewer";
import { Tabs } from "@/components/Tabs";
import { Config } from "@/lib/api";

import { MapSection } from "./MapSection";

export const ExploreSimulations: FC<{ cfg: Config }> = ({ cfg }) => {
  const simulationsSchema = z.array(
    z.object({
      name: z.string(),
      env: z.unknown(),
    })
  );

  const simulations = simulationsSchema.parse(cfg.config.value);

  const [selectedSimulation, setSelectedSimulation] = useState<string | null>(
    simulations[0]?.name ?? null
  );

  const selectedSimulationConfig = useMemo(() => {
    return simulationsSchema
      .parse(cfg.config.value)
      .find((s) => s.name === selectedSimulation);
  }, [cfg.config.value, selectedSimulation]);

  return (
    <div className="flex gap-8">
      <div>
        <h2 className="mb-2 ml-4 font-semibold">Simulations</h2>
        <div className="flex flex-col">
          {simulations.map((simulation) => (
            <div
              key={simulation.name}
              className={clsx(
                "cursor-pointer rounded-md px-4 py-1 text-sm",
                selectedSimulation === simulation.name
                  ? "bg-blue-200"
                  : "hover:bg-blue-100"
              )}
              onClick={() => setSelectedSimulation(simulation.name)}
            >
              {simulation.name}
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1">
        {selectedSimulation && selectedSimulationConfig ? (
          <Tabs
            topPadding
            tabs={[
              {
                id: "map",
                label: "Map",
                content: (
                  <MapSection
                    key={selectedSimulation}
                    cfg={cfg}
                    name={selectedSimulation!}
                  />
                ),
              },
              {
                id: "config",
                label: "Simulation Config",
                content: (
                  <ConfigViewer
                    key={selectedSimulation}
                    value={selectedSimulationConfig.env}
                  />
                ),
              },
            ]}
          ></Tabs>
        ) : (
          <div className="pt-4 text-center text-gray-500">
            No simulation selected
          </div>
        )}
      </div>
    </div>
  );
};
