import clsx from "clsx";
import Link from "next/link";
import { FC, PropsWithChildren, ReactNode } from "react";

export type LinkTab = {
  id: string;
  label: string;
  href: string;
  isActive?: boolean;
};

type LinkTabsProps = PropsWithChildren<{
  tabs: LinkTab[];
  activeTab?: string;
  additionalTabBarContent?: ReactNode;
  className?: string;
}>;

export const LinkTabs: FC<LinkTabsProps> = ({
  tabs,
  additionalTabBarContent,
  className,
  children,
  activeTab,
}) => {
  return (
    <div className={clsx("flex flex-1 flex-col", className)}>
      {/* Tab Bar */}
      <div className="flex border-b border-gray-300">
        {tabs.map((tab) => (
          <Link
            key={tab.id}
            href={tab.href}
            className={clsx(
              "cursor-pointer px-4 py-2 text-sm font-medium",
              activeTab === tab.id
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-500 hover:text-gray-900"
            )}
          >
            {tab.label}
          </Link>
        ))}
        {additionalTabBarContent && (
          <div className="ml-4 flex flex-1 items-center">
            {additionalTabBarContent}
          </div>
        )}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">{children}</div>
    </div>
  );
};
