import clsx from "clsx";
import { type FC, useMemo, useState } from "react";

import { Button } from "@/components/Button";
import { Dropdown } from "@/components/Dropdown";
import { DropdownMenu } from "@/components/Dropdown/DropdownMenu";
import { DropdownMenuActionItem } from "@/components/Dropdown/DropdownMenuActionItem";
import { FilterInput } from "@/components/FilterInput";
import { CheckIcon } from "@/components/icons/CheckIcon";
import { EmptyIcon } from "@/components/icons/EmptyIcon";
import { NoResultsMessage } from "@/components/NoResultsMessage";
import { Tooltip } from "@/components/Tooltip";
import { Variant } from "@/lib/api/cogames";

import { useVariants } from "./useVariants";

const VariantsMenu: FC<{
  allVariants: Variant[];
  selectedVariants: string[] | null;
  toggleVariant: (variant: string) => void;
}> = ({ allVariants, selectedVariants, toggleVariant }) => {
  const isSelected = (variant: string) => selectedVariants?.includes(variant);
  const [filter, setFilter] = useState("");
  const [focus, setFocus] = useState(true);

  const filtered = useMemo(() => {
    if (!filter) return allVariants;
    return allVariants.filter((variant) =>
      variant.name.toLowerCase().includes(filter.toLowerCase())
    );
  }, [allVariants, filter]);

  return (
    <DropdownMenu>
      <FilterInput
        placeholder="Filter variants..."
        focus={focus}
        value={filter}
        onBlur={() => setFocus(false)}
        onChange={setFilter}
      />

      <div className="max-h-120 overflow-y-auto">
        <NoResultsMessage className="px-4 py-2" show={filtered.length === 0} />

        {filtered.map((variant) => (
          <Tooltip
            key={variant.name}
            render={() => <div className="text-xs">{variant.description}</div>}
            placement="left"
          >
            <DropdownMenuActionItem
              key={variant.name}
              onClick={() => toggleVariant(variant.name)}
              icon={isSelected(variant.name) ? CheckIcon : EmptyIcon}
              title={variant.name}
            />
          </Tooltip>
        ))}
      </div>
    </DropdownMenu>
  );
};

export const VariantsSelector: FC<{ allVariants: Variant[] }> = ({
  allVariants,
}) => {
  const [variants, setVariants] = useVariants();

  const toggleVariant = (variant: string) => {
    if (variants.includes(variant)) {
      setVariants(variants?.filter((v) => v !== variant) ?? []);
    } else {
      setVariants([...(variants ?? []), variant]);
    }
  };

  return (
    <div className="flex w-full items-center justify-end gap-1.5">
      <div className="text-sm font-medium text-gray-600">Variants:</div>
      <Dropdown
        render={() => (
          <VariantsMenu
            allVariants={allVariants}
            selectedVariants={variants}
            toggleVariant={toggleVariant}
          />
        )}
      >
        <div
          className={clsx(
            "cursor-pointer rounded border px-2 py-1 text-xs",
            variants.length
              ? "border-blue-200 bg-blue-50 hover:bg-blue-100"
              : "border-gray-300 bg-gray-100 hover:bg-gray-200"
          )}
        >
          {variants.length ? variants.join(", ") : "none"}
        </div>
      </Dropdown>
      {variants.length ? (
        <Button theme="secondary" size="sm" onClick={() => setVariants([])}>
          x
        </Button>
      ) : null}
    </div>
  );
};
