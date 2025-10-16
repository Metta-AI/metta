"use client";

import React, { useState } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";

import { cn } from "@/lib/utils";

interface DataTableProps<TData> {
  data: TData[];
  columns: ColumnDef<TData, unknown>[];
  isLoading?: boolean;
  loadingMessage?: string;
  emptyMessage?: string;
  className?: string;
  initialSorting?: SortingState;
  onRowClick?: (row: TData) => void;
}

export function DataTable<TData>({
  data,
  columns,
  isLoading = false,
  loadingMessage = "Loading…",
  emptyMessage = "No results found.",
  className,
  initialSorting = [],
  onRowClick,
}: DataTableProps<TData>) {
  const [sorting, setSorting] = useState<SortingState>(initialSorting);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const columnCount = table.getAllColumns().length;

  return (
    <div
      className={cn(
        "border-border bg-card overflow-hidden rounded-xl border shadow-sm",
        className
      )}
    >
      <table className="divide-border min-w-full divide-y">
        <thead className="bg-muted/40">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  colSpan={header.colSpan}
                  className="text-muted-foreground px-4 py-3 text-left text-xs font-semibold tracking-wide uppercase"
                >
                  {header.isPlaceholder ? null : (
                    <button
                      type="button"
                      className={cn(
                        "text-muted-foreground hover:text-foreground flex items-center gap-1 transition",
                        header.column.getCanSort() ? "" : "cursor-default"
                      )}
                      onClick={header.column.getToggleSortingHandler()}
                      disabled={!header.column.getCanSort()}
                    >
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                      {header.column.getCanSort() && (
                        <span className="text-xs">
                          {{
                            asc: "↑",
                            desc: "↓",
                          }[header.column.getIsSorted() as string] ?? "↕"}
                        </span>
                      )}
                    </button>
                  )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="divide-border bg-card divide-y">
          {isLoading ? (
            <tr>
              <td
                colSpan={columnCount}
                className="text-muted-foreground px-4 py-8 text-center text-sm"
              >
                {loadingMessage}
              </td>
            </tr>
          ) : table.getRowModel().rows.length === 0 ? (
            <tr>
              <td
                colSpan={columnCount}
                className="text-muted-foreground px-4 py-12 text-center text-sm"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className={cn(
                  "hover:bg-muted/40 transition-colors",
                  onRowClick ? "cursor-pointer" : ""
                )}
                onClick={() => onRowClick?.(row.original)}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-3 align-top text-sm">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export type { ColumnDef } from "@tanstack/react-table";
