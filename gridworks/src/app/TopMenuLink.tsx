"use client";
import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { FC } from "react";

export const TopMenuLink: FC<{ href: string; children: React.ReactNode }> = ({
  href,
  children,
}) => {
  const pathname = usePathname();
  const isSelected = pathname === href;

  return (
    <Link
      href={href}
      className={clsx(
        "text-sm font-medium text-gray-500 hover:text-gray-900",
        isSelected && "text-gray-900"
      )}
    >
      {children}
    </Link>
  );
};
