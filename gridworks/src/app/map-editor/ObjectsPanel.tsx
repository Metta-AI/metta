import clsx from "clsx";
import { createContext, FC, use, useEffect, useRef } from "react";

import { useDrawer } from "@/components/MapViewer/hooks";
import { Tooltip } from "@/components/Tooltip";
import { Shortcut, useGlobalShortcuts } from "@/hooks/useGlobalShortcut";
import { Drawer } from "@/lib/draw/Drawer";
import { MAP_BACKGROUND_COLOR } from "@/lib/draw/drawGrid";
import { gridObjectRegistry } from "@/lib/gridObjectRegistry";

const ObjectsPanelContext = createContext<{
  enableHotkeys: boolean;
}>({
  enableHotkeys: false,
});

function useObjectShortcuts(
  setSelectedEntity: (entity: string) => void,
  selectedEntity: string,
  enableHotkeys: boolean
) {
  useGlobalShortcuts(
    enableHotkeys
      ? gridObjectRegistry.allHotkeys().map((hotkey) => {
          const shortcut: Shortcut = { key: hotkey };
          return [
            shortcut,
            () => {
              const object = gridObjectRegistry.objectByHotkey(
                hotkey,
                selectedEntity
              );
              if (!object) {
                return;
              }
              setSelectedEntity(object.name);
              const activeElement = document.activeElement;
              if (activeElement instanceof HTMLElement) {
                activeElement.blur();
              }
            },
          ];
        })
      : []
  );
}

const SelectableButton: FC<{
  children: React.ReactNode;
  isSelected: boolean;
  onClick: () => void;
  name: string;
}> = ({ children, isSelected, onClick, name }) => {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "w-full cursor-pointer",
        isSelected
          ? "bg-blue-100 ring-2 ring-blue-300"
          : "hover:ring-2 hover:ring-blue-300"
      )}
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

    canvas.width = canvas.height = scaledSize;

    drawer.drawObject(name, ctx, 0, 0, scaledSize);
  }, [name, drawer]);

  let content: React.ReactNode;
  if (name === "empty") {
    content = (
      <div
        className="h-8 w-8"
        style={{ backgroundColor: MAP_BACKGROUND_COLOR }}
      />
    );
  } else {
    content = (
      <canvas
        style={{ width: size, height: size }}
        ref={canvasRef}
        className="rounded-sm"
      />
    );
  }

  const { enableHotkeys } = use(ObjectsPanelContext);

  return (
    <Tooltip
      render={() => (
        <div>
          <header>{name}</header>
          {enableHotkeys && (
            <div>Hotkey: {gridObjectRegistry.objectByName(name)?.hotkey}</div>
          )}
        </div>
      )}
    >
      {content}
    </Tooltip>
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
      name={name}
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
  drawer: Drawer;
}> = ({ groupName, names, selected, onClick, drawer }) => {
  return (
    <div className={clsx("flex items-center justify-between gap-2")}>
      <div className="mx-1 font-mono text-xs tracking-wider text-gray-600 uppercase">
        {groupName}
      </div>
      <div className="flex items-stretch">
        {names.map((name) => (
          <SelectableButton
            key={name}
            onClick={() => onClick(name)}
            isSelected={selected === name}
            name={name}
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
  enableHotkeys?: boolean;
}> = ({ selectedEntity, setSelectedEntity, enableHotkeys = false }) => {
  const drawer = useDrawer();

  useObjectShortcuts(setSelectedEntity, selectedEntity, enableHotkeys);

  if (!drawer) {
    return null;
  }

  const basicNames: string[] = [];
  const groupedNames: Record<string, string[]> = {};

  for (const name of gridObjectRegistry.objectNames) {
    if (
      name.startsWith("agent.") ||
      name.startsWith("mine_") ||
      name.startsWith("generator_")
    ) {
      const groupName = name.startsWith("agent.")
        ? "agent"
        : name.split("_")[0];
      groupedNames[groupName] = [...(groupedNames[groupName] || []), name];
    } else {
      basicNames.push(name);
    }
  }

  return (
    <ObjectsPanelContext.Provider value={{ enableHotkeys }}>
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
    </ObjectsPanelContext.Provider>
  );
};
