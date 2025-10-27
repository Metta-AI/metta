import React from "react";
import { Button } from "@/components/ui/button";

interface SortOption {
  key: string;
  label: string;
}

interface SortControlsProps {
  sortOptions: SortOption[];
  sortBy: string;
  sortDirection: "asc" | "desc";
  onSortChange: (key: string) => void;
  onDirectionToggle: () => void;
  className?: string;
}

export function SortControls({
  sortOptions,
  sortBy,
  sortDirection,
  onSortChange,
  onDirectionToggle,
  className,
}: SortControlsProps) {
  return (
    <div
      className={`flex flex-col gap-3 md:flex-row md:items-center md:justify-between ${className || ""}`}
    >
      <span className="text-sm font-medium text-gray-600">Sort by:</span>
      <div className="flex flex-wrap gap-2">
        {sortOptions.map((option) => {
          const isActive = sortBy === option.key;
          return (
            <Button
              key={option.key}
              variant={isActive ? "default" : "outline"}
              size="sm"
              onClick={() => {
                if (isActive) {
                  onDirectionToggle();
                } else {
                  onSortChange(option.key);
                }
              }}
            >
              {option.label}
            </Button>
          );
        })}
      </div>
    </div>
  );
}
