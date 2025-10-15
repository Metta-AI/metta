import React from "react";
import { cn } from "@/lib/utils";

interface Tab {
  id: string;
  label: React.ReactNode;
  disabled?: boolean;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  className?: string;
}

export function Tabs({ tabs, activeTab, onTabChange, className }: TabsProps) {
  return (
    <div className={cn("flex gap-2", className)}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => !tab.disabled && onTabChange(tab.id)}
          disabled={tab.disabled}
          className={cn(
            "cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
            activeTab === tab.id
              ? "bg-blue-100 text-blue-700"
              : "text-gray-600 hover:bg-gray-100",
            tab.disabled && "cursor-not-allowed opacity-50"
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

