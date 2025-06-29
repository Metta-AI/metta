import { createContext, FC, useContext } from "react";

import { SceneTree } from "@/lib/api";

import { JsonAsYaml } from "./JsonAsYaml";

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

type TreeProps = {
  sceneTree: SceneTree;
};

type ContextProps = {
  onSceneSelect: (sceneTree: SceneTree | undefined) => void;
};

const SceneTreeViewerContext = createContext<ContextProps>({
  onSceneSelect: () => {},
});

const InnerSceneTreeViewer: FC<TreeProps> = ({ sceneTree }) => {
  const { onSceneSelect } = useContext(SceneTreeViewerContext);

  return (
    <div className="overflow-auto font-mono">
      <div
        className="cursor-pointer text-sm font-bold"
        onMouseEnter={() => onSceneSelect(sceneTree)}
        onMouseLeave={() => onSceneSelect(undefined)}
      >
        {sceneTree.type}
        <span className="ml-1 text-xs text-gray-500">
          ({sceneTree.area.x}, {sceneTree.area.y}) {sceneTree.area.width}x
          {sceneTree.area.height}
        </span>
      </div>
      <JsonAsYaml json={sceneTree.params} />
      {sceneTree.children.length > 0 && (
        <div>
          <div className="text-xs font-bold">Children:</div>
          <div className="pl-4">
            {sceneTree.children.map((child, i) => (
              <InnerSceneTreeViewer key={i} sceneTree={child} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export const SceneTreeViewer: FC<TreeProps & ContextProps> = ({
  sceneTree,
  ...contextProps
}) => {
  return (
    <SceneTreeViewerContext.Provider value={contextProps}>
      <InnerSceneTreeViewer sceneTree={sceneTree} />
    </SceneTreeViewerContext.Provider>
  );
};
