import { FC, use } from "react";

import { CornerMenu } from "../CornerMenu";
import { useCloseDropdown } from "../Dropdown/DropdownContext";
import { DropdownMenuActionItem } from "../Dropdown/DropdownMenuActionItem";
import { CheckIcon } from "../icons/CheckIcon";
import { EmptyIcon } from "../icons/EmptyIcon";
import { CopyIcon } from "../icons/CopyIcon";
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

const ToggleDefaultValuesItem: FC = () => {
  const closeDropdown = useCloseDropdown();
  const { showDefaultValues, setShowDefaultValues } = use(YamlContext);
  return (
    <DropdownMenuActionItem
      title="Show Default Values"
      icon={showDefaultValues ? CheckIcon : EmptyIcon}
      onClick={() => {
        setShowDefaultValues(!showDefaultValues);
        closeDropdown();
      }}
    />
  );
};

const CopyAsYamlItem: FC<{
  onCopyAsYaml?: () => void;
}> = ({ onCopyAsYaml }) => {
  const closeDropdown = useCloseDropdown();
  return (
    <DropdownMenuActionItem
      title="Copy as YAML"
      icon={CopyIcon}
      onClick={() => {
        onCopyAsYaml?.();
        closeDropdown();
      }}
    />
  );
};

export const Menu: FC<{
  onCopyAsYaml?: () => void;
}> = ({ onCopyAsYaml }) => {
  return (
    <CornerMenu
      renderItems={() => (
        <>
          <ToggleDefaultValuesItem />
          <ToggleDebugItem />
          <CopyAsYamlItem onCopyAsYaml={onCopyAsYaml} />
        </>
      )}
    />
  );
};
