"use client";

import React, { useMemo } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";

import type { PaperSummary } from "@/lib/api/resources/papers";
import { cn } from "@/lib/utils";

interface PapersTableProps {
  papers: PaperSummary[];
  isLoading?: boolean;
  onRowClick?: (paper: PaperSummary) => void;
  onTagClick?: (tag: string) => void;
}

export function PapersTable({
  papers,
  isLoading = false,
  onRowClick,
  onTagClick,
}: PapersTableProps) {
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: "createdAt", desc: true },
  ]);

  const columns = useMemo<ColumnDef<PaperSummary>[]>(
    () => [
      {
        accessorKey: "title",
        header: "Title",
        cell: ({ row }) => {
          const paper = row.original;
          return (
            <div className="space-y-1">
              <p className="font-medium text-gray-900">{paper.title}</p>
              {paper.tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {paper.tags.slice(0, 3).map((tag) => (
                    <button
                      key={tag}
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        onTagClick?.(tag);
                      }}
                      className="rounded bg-blue-50 px-2 py-0.5 text-xs text-blue-700 transition hover:bg-blue-100"
                    >
                      {tag}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        },
      },
      {
        accessorKey: "author",
        header: "Authors",
        cell: ({ row }) => (
          <div className="space-y-1 text-sm text-gray-600">
            {row.original.authors.slice(0, 3).map((author) => (
              <div key={author.id}>{author.name}</div>
            ))}
            {row.original.authors.length > 3 && (
              <div className="text-xs text-gray-400">
                +{row.original.authors.length - 3} more
              </div>
            )}
          </div>
        ),
      },
      {
        accessorKey: "stars",
        header: () => <span className="flex items-center gap-1">Stars</span>,
        cell: ({ row }) => (
          <div className="text-sm font-medium text-gray-900">
            {row.original.stars}
          </div>
        ),
      },
      {
        accessorKey: "createdAt",
        header: "Published",
        cell: ({ row }) => (
          <div className="text-sm text-gray-600">
            {new Date(row.original.createdAt).toLocaleDateString()}
          </div>
        ),
      },
    ],
    []
  );

  const table = useReactTable({
    data: papers,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="overflow-hidden rounded-xl border border-gray-200 shadow-sm">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  colSpan={header.colSpan}
                  className="px-4 py-3 text-left text-xs font-semibold tracking-wide text-gray-500 uppercase"
                >
                  {header.isPlaceholder ? null : (
                    <button
                      type="button"
                      className="flex items-center gap-1 text-gray-700 hover:text-gray-900"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                      {{
                        asc: "↑",
                        desc: "↓",
                      }[header.column.getIsSorted() as string] ?? null}
                    </button>
                  )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {isLoading ? (
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-6 text-center text-sm text-gray-500"
              >
                Loading papers…
              </td>
            </tr>
          ) : (
            table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className={cn(
                  "hover:bg-blue-50/60",
                  onRowClick ? "cursor-pointer" : ""
                )}
                onClick={() => onRowClick?.(row.original)}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-3 align-top">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))
          )}
          {table.getRowModel().rows.length === 0 && !isLoading && (
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-10 text-center text-sm text-gray-500"
              >
                No papers found.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
