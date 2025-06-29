import { FC } from "react";

import { SceneTree } from "@/lib/api";

const ParamValue: FC<{ value: unknown }> = ({ value }) => {
  if (typeof value === "object" && value !== null) {
    return <pre className="text-xs">{JSON.stringify(value, null, 2)}</pre>;
  }
  return <span>{String(value)}</span>;
};

const ParamsViewer: FC<{ params: Record<string, unknown> }> = ({ params }) => {
  return (
    <div className="overflow-auto">
      {Object.entries(params).map(([key, value]) => (
        <div key={key} className="text-xs">
          <span className="font-bold text-gray-600">{key}:</span>{" "}
          <ParamValue value={value} />
        </div>
      ))}
    </div>
  );
};

export const SceneTreeViewer: FC<{ sceneTree: SceneTree }> = ({
  sceneTree,
}) => {
  return (
    <div className="overflow-auto font-mono">
      <div className="text-sm font-bold">
        {sceneTree.type} ({sceneTree.area.x}, {sceneTree.area.y}){" "}
        {sceneTree.area.width}x{sceneTree.area.height}
      </div>
      <ParamsViewer params={sceneTree.params} />
      {sceneTree.children.length > 0 && (
        <div>
          <div className="text-xs font-bold">Children:</div>
          <div className="pl-4">
            {sceneTree.children.map((child) => (
              <SceneTreeViewer key={child.type} sceneTree={child} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
