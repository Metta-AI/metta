"use client";
import yaml from "js-yaml";
import { useQueryState } from "nuqs";
import { FC, useMemo } from "react";

import { FilterItem, parseFilterParam } from "@/app/stored-maps/dir/params";
import { MettaGrid } from "@/lib/MettaGrid";
import { MapData } from "@/server/types";

import { CopyToClipboardButton } from "./CopyToClipboardButton";
import { JsonAsYaml } from "./JsonAsYaml";
import { MapViewer } from "./MapViewer";

// YAML viewer with the ability to click lines to filter the map list
const FrontmatterViewer: FC<{ frontmatter: Record<string, unknown> }> = ({
  frontmatter,
}) => {
  const [filters, setFilters] = useQueryState(
    "filter",
    parseFilterParam.withOptions({ shallow: false })
  );

  // Function to check if a key-value pair is currently in the filters
  const isFiltered = (key: string, value: string) => {
    return (
      filters?.some((filter) => filter.key === key && filter.value === value) ||
      false
    );
  };

  // Function to handle clicking on a frontmatter line
  const handleSelectLine = (key: string, value: string) => {
    const newFilter: FilterItem = { key, value };

    // Check if this filter already exists
    const filterExists =
      filters?.some((filter) => filter.key === key && filter.value === value) ||
      false;

    if (filterExists) {
      // Remove the filter if it already exists
      setFilters(
        filters?.filter(
          (filter) => !(filter.key === key && filter.value === value)
        ) || []
      );
    } else {
      // Add the new filter
      setFilters([...(filters || []), newFilter]);
    }
  };

  return (
    <JsonAsYaml
      json={frontmatter}
      isSelected={isFiltered}
      onSelectLine={handleSelectLine}
    />
  );
};

export const ExtendedMapViewer: FC<{ mapData: MapData }> = ({ mapData }) => {
  // Parse the frontmatter YAML
  const frontmatterJson = useMemo(() => {
    try {
      return yaml.load(mapData.content.frontmatter) as Record<string, unknown>;
    } catch (error) {
      console.error("Error parsing frontmatter:", error);
      return {};
    }
  }, [mapData.content.frontmatter]);

  const grid = useMemo(
    () => MettaGrid.fromAscii(mapData.content.data),
    [mapData.content.data]
  );

  return (
    <div className="grid grid-cols-[400px_1fr_250px] gap-8">
      <div className="max-h-[80vh] overflow-auto">
        <FrontmatterViewer frontmatter={frontmatterJson} />
      </div>
      <div className="flex flex-col items-center justify-start overflow-auto">
        <div className="max-w-full">
          <MapViewer grid={grid} />
        </div>
      </div>
      <CopyToClipboardButton text={mapData.content.data}>
        Copy Map Data to Clipboard
      </CopyToClipboardButton>
    </div>
  );
};
