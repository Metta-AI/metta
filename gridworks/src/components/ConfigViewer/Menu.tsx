import { FC, use } from "react";

import { Dropdown } from "../Dropdown";
import { useCloseDropdown } from "../Dropdown/DropdownContext";
import { DropdownMenu } from "../Dropdown/DropdownMenu";
import { DropdownMenuActionItem } from "../Dropdown/DropdownMenuActionItem";
import { CheckIcon } from "../icons/CheckIcon";
import { EmptyIcon } from "../icons/EmptyIcon";
import { YamlContext } from "./YamlContext";

const ToggleDebugItem: FC = () => {
  const closeDropdown = useCloseDropdown();
  const { showDebugInfo, setShowDebugInfo } = use(YamlContext);
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

export const Menu: FC = () => {
  return (
    <Dropdown
      render={() => (
        <DropdownMenu>
          <ToggleDebugItem />
        </DropdownMenu>
      )}
    >
      <div className="h-4 w-4 cursor-pointer rounded bg-gray-200 text-center text-gray-700 hover:bg-gray-300">
        ▼
      </div>
    </Dropdown>
  );
};
