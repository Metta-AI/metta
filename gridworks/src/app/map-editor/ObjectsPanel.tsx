import clsx from "clsx";
import { FC, useEffect, useRef } from "react";

import { useDrawer } from "@/components/MapViewer/hooks";
import { Drawer, objectNames } from "@/lib/draw/Drawer";

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

    drawer.drawObject(name, ctx, 0, 0, scaledSize);
  }, [name, drawer]);

  if (name === "empty") {
    return <div className="h-8 w-8 bg-gray-200" />;
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
    <div className={clsx("flex items-center justify-between gap-2")}>
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

  const basicNames: string[] = [];
  const groupedNames: Record<string, string[]> = {};

  for (const name of objectNames) {
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
