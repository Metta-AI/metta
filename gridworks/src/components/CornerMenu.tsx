import { FC, ReactNode } from "react";

import { Dropdown } from "./Dropdown";
import { DropdownMenu } from "./Dropdown/DropdownMenu";

export const CornerMenu: FC<{ renderItems: () => ReactNode }> = ({
  renderItems,
}) => {
  return (
    <Dropdown render={() => <DropdownMenu>{renderItems()}</DropdownMenu>}>
      <div className="h-4 w-4 cursor-pointer rounded bg-gray-200 text-center text-xs text-gray-700 hover:bg-gray-300">
        ▼
      </div>
    </Dropdown>
  );
};
