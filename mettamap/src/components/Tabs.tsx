"use client";
import clsx from "clsx";
import { FC, ReactNode, useState } from "react";

export type Tab = {
  id: string;
  label: string;
  content: ReactNode;
};

type TabsProps = {
  tabs: Tab[];
  defaultTab?: string;
  additionalTabBarContent?: ReactNode;
  className?: string;
};

export const Tabs: FC<TabsProps> = ({
  tabs,
  defaultTab,
  additionalTabBarContent,
  className,
}) => {
  const [activeTab, setActiveTab] = useState(
    defaultTab || (tabs.length > 0 ? tabs[0].id : "")
  );

  const activeTabContent = tabs.find((tab) => tab.id === activeTab)?.content;

  return (
    <div className={clsx("flex flex-1 flex-col", className)}>
      {/* Tab Bar */}
      <div className="flex border-b border-gray-300">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={clsx(
              "cursor-pointer px-4 py-2 text-sm font-medium",
              activeTab === tab.id
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-500 hover:text-gray-900"
            )}
          >
            {tab.label}
          </button>
        ))}
        {additionalTabBarContent && (
          <div className="ml-4 flex items-center">
            {additionalTabBarContent}
          </div>
        )}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">{activeTabContent}</div>
    </div>
  );
};
