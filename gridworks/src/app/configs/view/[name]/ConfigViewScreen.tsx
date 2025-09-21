import { FC } from "react";

import { ConfigViewer } from "@/components/ConfigViewer";
import { Tab, Tabs } from "@/components/Tabs";
import { Config } from "@/lib/api";

import { ExploreSimulations } from "./ExploreSimulations";
import { MapSection } from "./MapSection";
import { RunToolSection } from "./RunToolSection";

export const ConfigViewScreen: FC<{ cfg: Config }> = ({ cfg }) => {
  const tabs: Tab[] = [];

  if (cfg.maker.kind === "List[SimulationConfig]") {
    tabs.push({
      id: "explore",
      label: "Explore",
      content: (
        <div className="pt-4">
          <ExploreSimulations cfg={cfg} />
        </div>
      ),
    });
  }

  tabs.push({
    id: "config",
    label: "Config",
    content: (
      <div className="pt-4">
        <ConfigViewer
          value={cfg.config.value}
          unsetFields={cfg.config.unset_fields}
        />
      </div>
    ),
  });

  if (
    [
      "MettaGridConfig",
      "CurriculumConfig",
      "PlayTool",
      "ReplayTool",
      "TrainTool",
    ].includes(cfg.maker.kind)
  ) {
    tabs.push({
      id: "map",
      label: "Map",
      content: (
        <div className="pt-4">
          <MapSection cfg={cfg} />
        </div>
      ),
    });
  }

  if (cfg.maker.kind.endsWith("Tool")) {
    tabs.push({
      id: "run",
      label: "Run",
      content: <RunToolSection cfg={cfg} />,
    });
  }

  return (
    <div>
      <Tabs tabs={tabs} defaultTab="config" />
    </div>
  );
};
