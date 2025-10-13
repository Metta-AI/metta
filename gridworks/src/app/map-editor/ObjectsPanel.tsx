import clsx from "clsx";
import { FC, useEffect, useRef } from "react";

import { useDrawer } from "@/components/MapViewer/hooks";
import { BACKGROUND_MAP_COLOR, Drawer, objectNames } from "@/lib/draw/Drawer";
import { MettaObject } from "@/lib/MettaGrid";

const SelectableButton: FC<{
  children: React.ReactNode;
  isSelected: boolean;
  onClick: () => void;
  title: string;
}> = ({ children, isSelected, onClick, title }) => {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "cursor-pointer",
        isSelected
          ? "bg-blue-100 ring-2 ring-blue-300"
          : "hover:ring-2 hover:ring-blue-300"
      )}
      title={title}
    >
      {children}
    </button>
  );
};

const ObjectIcon: FC<{
  name: string;
  drawer: Drawer;
}> = ({ name, drawer }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const size = 32;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    const scaledSize = size * dpr;

    canvas.width = scaledSize;
    canvas.height = scaledSize;

    ctx.scale(scaledSize, scaledSize);

    drawer.drawObject(ctx, MettaObject.fromObjectName(0, 0, name)!);
  }, [name, drawer]);

  if (name === "empty") {
    return (
      <div
        className="h-8 w-8"
        style={{ backgroundColor: BACKGROUND_MAP_COLOR }}
      />
    );
  }

  return (
    <canvas
      style={{ width: size, height: size }}
      ref={canvasRef}
      className="rounded-sm"
    />
  );
};

const ObjectEntry: FC<{
  name: string;
  onClick: () => void;
  isSelected: boolean;
  drawer: Drawer;
}> = ({ name, onClick, isSelected, drawer }) => {
  return (
    <SelectableButton
      onClick={() => onClick()}
      title={name}
      isSelected={isSelected}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="mx-1 font-mono text-xs tracking-wider text-gray-600 uppercase">
          {name}
        </div>
        <ObjectIcon name={name} drawer={drawer} />
      </div>
    </SelectableButton>
  );
};

const GroupedObjectEntry: FC<{
  groupName: string;
  names: string[];
  selected: string;
  onClick: (name: string) => void;
  // isSelected: boolean;
  drawer: Drawer;
}> = ({ groupName, names, selected, onClick, drawer }) => {
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="mx-1 font-mono text-xs tracking-wider text-gray-600 uppercase">
        {groupName}
      </div>
      <div className="flex">
        {names.map((name) => (
          <SelectableButton
            key={name}
            onClick={() => onClick(name)}
            isSelected={selected === name}
            title={name}
          >
            <ObjectIcon name={name} drawer={drawer} />
          </SelectableButton>
        ))}
      </div>
    </div>
  );
};

export const ObjectsPanel: FC<{
  selectedEntity: string;
  setSelectedEntity: (entity: string) => void;
}> = ({ selectedEntity, setSelectedEntity }) => {
  const drawer = useDrawer();

  if (!drawer) {
    return null;
  }

  const basicNames: string[] = ["empty", "wall"];
  const groupedNames: Record<string, string[]> = {};

  const groupRegexes = {
    agent: /agent\.(\w+)/,
    mine: /mine_(\w+)/,
    generator: /generator_(\w+)/,
    chest: /chest_(\w+)/,
    clipped_extractor: /clipped_(\w+)_extractor/,
    extractor: /(\w+)_extractor/,
    ex_dep: /(\w+)_ex_dep/,
  };

  for (const name of objectNames) {
    if (name === "empty" || name === "wall") {
      continue;
    }

    let isGrouped = false;
    for (const [groupName, regex] of Object.entries(groupRegexes)) {
      const match = name.match(regex);
      if (match) {
        groupedNames[groupName] = [...(groupedNames[groupName] || []), name];
        isGrouped = true;
        break;
      }
    }
    if (!isGrouped) {
      basicNames.push(name);
    }
  }

  return (
    <div className="flex flex-col gap-1">
      {basicNames.map((key) => (
        <ObjectEntry
          key={key}
          name={key}
          onClick={() => setSelectedEntity(key)}
          isSelected={selectedEntity === key}
          drawer={drawer}
        />
      ))}
      {Object.entries(groupedNames).map(([key, names]) => (
        <GroupedObjectEntry
          key={key}
          groupName={key}
          names={names}
          selected={selectedEntity}
          onClick={setSelectedEntity}
          drawer={drawer}
        />
      ))}
    </div>
  );
};
