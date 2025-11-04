import { FC, use } from "react";

import { CornerMenu } from "../CornerMenu";
import { useCloseDropdown } from "../Dropdown/DropdownContext";
import { DropdownMenuActionItem } from "../Dropdown/DropdownMenuActionItem";
import { CheckIcon } from "../icons/CheckIcon";
import { EmptyIcon } from "../icons/EmptyIcon";
import { MapViewerContext } from "./MapViewerContext";

const ToggleDebugItem: FC = () => {
  const closeDropdown = useCloseDropdown();
  const { showDebugInfo, setShowDebugInfo } = use(MapViewerContext);
  return (
    <DropdownMenuActionItem
      title="Show Debug Info"
      icon={showDebugInfo ? CheckIcon : EmptyIcon}
      onClick={() => {
        setShowDebugInfo(!showDebugInfo);
        closeDropdown();
      }}
    />
  );
};

const ToggleHoverInfoItem: FC = () => {
  const closeDropdown = useCloseDropdown();
  const { showHoverInfo, setShowHoverInfo } = use(MapViewerContext);
  return (
    <DropdownMenuActionItem
      title="Show Hover Info"
      icon={showHoverInfo ? CheckIcon : EmptyIcon}
      onClick={() => {
        setShowHoverInfo(!showHoverInfo);
        closeDropdown();
      }}
    />
  );
};

export const MapViewerCornerMenu = () => {
  return (
    <CornerMenu
      renderItems={() => (
        <>
          <ToggleDebugItem />
          <ToggleHoverInfoItem />
        </>
      )}
    />
  );
};
