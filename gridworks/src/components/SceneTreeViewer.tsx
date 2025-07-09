import clsx from "clsx";
import { createContext, FC, useContext, useState } from "react";

import { SceneTree } from "@/lib/api";

import { JsonAsYaml } from "./JsonAsYaml";

const ParamsViewer: FC<{ params: Record<string, unknown> }> = ({ params }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (Object.keys(params).length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex cursor-pointer items-center gap-1 text-xs font-medium text-blue-700 hover:text-blue-900"
      >
        <span
          className={clsx(
            "transform transition-transform",
            isExpanded ? "rotate-90" : ""
          )}
        >
          ▶
        </span>
        Parameters ({Object.keys(params).length})
      </button>
      {isExpanded && <JsonAsYaml json={params} />}
    </div>
  );
};

const ChildrenViewer: FC<{
  childrenScenes: SceneTree[];
  isExpanded: boolean;
  onToggle: () => void;
  depth: number;
}> = ({ childrenScenes, isExpanded, onToggle }) => {
  if (childrenScenes.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={onToggle}
        className="flex cursor-pointer items-center gap-1 text-xs font-medium text-blue-700 hover:text-blue-900"
      >
        <span
          className={clsx(
            "transform transition-transform",
            isExpanded ? "rotate-90" : ""
          )}
        >
          ▶
        </span>
        Children ({childrenScenes.length})
      </button>
    </div>
  );
};

function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash);
}

function getSceneTypeColor(type: string) {
  const hash = hashString(type);

  // Define a set of pleasant color combinations
  const colorPalettes = [
    "bg-blue-50 border-blue-200 text-blue-900",
    "bg-green-50 border-green-200 text-green-900",
    "bg-purple-50 border-purple-200 text-purple-900",
    "bg-orange-50 border-orange-200 text-orange-900",
    "bg-yellow-50 border-yellow-200 text-yellow-900",
    "bg-pink-50 border-pink-200 text-pink-900",
    "bg-indigo-50 border-indigo-200 text-indigo-900",
    "bg-red-50 border-red-200 text-red-900",
    "bg-teal-50 border-teal-200 text-teal-900",
    "bg-cyan-50 border-cyan-200 text-cyan-900",
    "bg-emerald-50 border-emerald-200 text-emerald-900",
    "bg-violet-50 border-violet-200 text-violet-900",
    "bg-rose-50 border-rose-200 text-rose-900",
    "bg-amber-50 border-amber-200 text-amber-900",
    "bg-lime-50 border-lime-200 text-lime-900",
    "bg-sky-50 border-sky-200 text-sky-900",
  ];

  return colorPalettes[hash % colorPalettes.length];
}

type TreeProps = {
  sceneTree: SceneTree;
  depth?: number;
};

type ContextProps = {
  onSceneSelect: (sceneTree: SceneTree | undefined) => void;
};

const SceneTreeViewerContext = createContext<ContextProps>({
  onSceneSelect: () => {},
});

const InnerSceneTreeViewer: FC<TreeProps> = ({ sceneTree, depth = 0 }) => {
  const { onSceneSelect } = useContext(SceneTreeViewerContext);
  const [isExpanded, setIsExpanded] = useState(true);
  const hasChildren = sceneTree.children.length > 0;

  const colorClass = getSceneTypeColor(sceneTree.type);

  return (
    <div className={clsx(depth > 0 && "mt-2 ml-4")}>
      <div
        className={clsx(
          "rounded-lg border p-3 transition-all duration-100 hover:brightness-90",
          colorClass
        )}
        onMouseEnter={() => onSceneSelect(sceneTree)}
        onMouseLeave={() => onSceneSelect(undefined)}
      >
        <div className="flex items-start justify-between">
          <div className="min-w-0 flex-1">
            <div className="mb-1 flex items-center gap-2 overflow-hidden">
              <span className="truncate text-sm font-bold">
                {sceneTree.type}
              </span>
              <div className="flex flex-shrink-0 items-center gap-1 rounded px-2 py-0.5 font-mono text-xs">
                <span className="text-gray-600">
                  ({sceneTree.area.x}, {sceneTree.area.y})
                </span>
                <span className="text-gray-400">•</span>
                <span className="font-medium text-gray-800">
                  {sceneTree.area.width}×{sceneTree.area.height}
                </span>
              </div>
            </div>

            <ParamsViewer params={sceneTree.params} />
            <ChildrenViewer
              childrenScenes={sceneTree.children}
              isExpanded={isExpanded}
              onToggle={() => setIsExpanded(!isExpanded)}
              depth={depth}
            />
          </div>
        </div>
      </div>

      {hasChildren && isExpanded && (
        <div className="mt-2 ml-2 border-l-2 border-gray-200 pl-2">
          {sceneTree.children.map((child, i) => (
            <InnerSceneTreeViewer key={i} sceneTree={child} depth={depth + 1} />
          ))}
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
    <div className="h-full overflow-auto pt-2">
      <SceneTreeViewerContext.Provider value={contextProps}>
        <div className="min-w-fit">
          <InnerSceneTreeViewer sceneTree={sceneTree} />
        </div>
      </SceneTreeViewerContext.Provider>
    </div>
  );
};
