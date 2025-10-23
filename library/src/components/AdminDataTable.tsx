"use client";

import React from "react";
import type { SortingState } from "@tanstack/react-table";

import { DataTable } from "@/components/ui/data-table";
import type { ColumnDef } from "@/components/ui/data-table";

interface AdminDataTableProps<TData> {
  data: TData[];
  columns: ColumnDef<TData, unknown>[];
  isLoading?: boolean;
  emptyMessage?: string;
  className?: string;
  initialSorting?: SortingState;
  onRowClick?: (row: TData) => void;
}

/**
 * AdminDataTable
 *
 * A wrapper around the generic DataTable component for admin pages.
 * Provides backward compatibility with existing admin components.
 */
export function AdminDataTable<TData>(props: AdminDataTableProps<TData>) {
  return <DataTable {...props} />;
}

export type { ColumnDef } from "@tanstack/react-table";
