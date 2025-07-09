"use client";
import { useQueryState } from "nuqs";
import { FC, useEffect, useState } from "react";
import Select from "react-select";

import { loadStoredMapIndex } from "@/lib/api";
import { MapIndex } from "@/server/types";

import { FilterItem, parseFilterParam } from "./params";

const FilterPair: FC<{
  filter: FilterItem;
}> = ({ filter }) => {
  return (
    <div className="font-mono text-xs">
      <span className="font-bold">{filter.key}</span> = {filter.value}
    </div>
  );
};

const DeleteButton: FC<{
  onClick: () => void;
}> = ({ onClick }) => {
  return (
    <button
      className="cursor-pointer rounded-md px-2 py-1 text-sm text-gray-700 hover:bg-red-50 hover:text-red-500"
      onClick={onClick}
    >
      x
    </button>
  );
};

const NewFilter: FC<{
  mapIndex: MapIndex;
  setFilter: (filter: FilterItem) => void;
  cancel: () => void;
}> = ({ mapIndex, setFilter, cancel }) => {
  const [state, setState] = useState<
    | {
        kind: "empty";
      }
    | {
        kind: "key";
        key: string;
      }
  >({
    kind: "empty",
  });

  if (state.kind === "empty") {
    return (
      <div className="flex items-center gap-2">
        <div className="flex-1">
          <Select
            autoFocus
            classNames={{
              control: () => "h-6",
            }}
            options={Object.keys(mapIndex).map((key) => ({
              label: key,
              value: key,
            }))}
            onChange={(v) => {
              if (!v?.value) {
                return;
              }
              setState({ kind: "key", key: v.value });
            }}
          />
        </div>
        <DeleteButton onClick={cancel} />
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <div className="font-mono text-xs">{state.key}</div>
      <div className="flex-1">
        <Select
          autoFocus
          classNames={{
            control: () => "h-6",
          }}
          options={Object.keys(mapIndex[state.key]).map((key) => ({
            label: key,
            value: key,
          }))}
          onChange={(v) => {
            if (!v?.value) {
              return;
            }
            setFilter({ key: state.key, value: v.value });
          }}
        />
      </div>
      <DeleteButton onClick={cancel} />
    </div>
  );
};

export const MapFilter: FC = () => {
  const [filters, setFilters] = useQueryState(
    "filter",
    parseFilterParam.withOptions({ shallow: false })
  );
  const [dir] = useQueryState("dir");

  const [newFilter, setNewFilter] = useState(false);

  const [mapIndex, setMapIndex] = useState<MapIndex | null>(null);
  useEffect(() => {
    if (dir) {
      loadStoredMapIndex(dir).then(setMapIndex);
    }
    return () => {
      setMapIndex(null);
    };
  }, [dir]);

  return (
    <div>
      {filters?.map((filter, i) => {
        return (
          <div key={i} className="flex items-center gap-2">
            <div className="flex-1">
              <FilterPair filter={filter} />
            </div>
            <DeleteButton
              onClick={() => {
                setFilters(filters.filter((_, j) => j !== i));
              }}
            />
          </div>
        );
      })}
      {newFilter && mapIndex ? (
        <NewFilter
          key={filters?.length ?? 0}
          mapIndex={mapIndex}
          setFilter={(v) => {
            setFilters([...(filters ?? []), v]);
            setNewFilter(false);
          }}
          cancel={() => {
            setNewFilter(false);
          }}
        />
      ) : (
        <button
          className="cursor-pointer rounded-md bg-blue-500 px-2 py-1 text-xs text-white disabled:bg-gray-300"
          disabled={!mapIndex}
          onClick={() => {
            setNewFilter(true);
          }}
        >
          Add Filter
        </button>
      )}
    </div>
  );
};
