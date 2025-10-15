"use client";

import React, { useMemo } from "react";

import type { PaperSummary } from "@/lib/api/resources/papers";
import { DataTable, type ColumnDef } from "@/components/ui/data-table";

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
  const columns = useMemo<ColumnDef<PaperSummary>[]>(
    () => [
      {
        accessorKey: "title",
        header: "Title",
        cell: ({ row }) => {
          const paper = row.original;
          return (
            <div className="space-y-1">
              <p className="font-medium">{paper.title}</p>
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
          <div className="text-muted-foreground space-y-0.5">
            {row.original.authors.slice(0, 3).map((author) => (
              <div key={author.id}>{author.name}</div>
            ))}
            {row.original.authors.length > 3 && (
              <div className="text-xs opacity-70">
                +{row.original.authors.length - 3} more
              </div>
            )}
          </div>
        ),
      },
      {
        accessorKey: "stars",
        header: "Stars",
        cell: ({ row }) => (
          <div className="font-medium">{row.original.stars}</div>
        ),
      },
      {
        accessorKey: "createdAt",
        header: "Published",
        cell: ({ row }) => (
          <div className="text-muted-foreground">
            {new Date(row.original.createdAt).toLocaleDateString()}
          </div>
        ),
      },
    ],
    [onTagClick]
  );

  return (
    <DataTable
      data={papers}
      columns={columns}
      isLoading={isLoading}
      loadingMessage="Loading papers…"
      emptyMessage="No papers found."
      initialSorting={[{ id: "createdAt", desc: true }]}
      onRowClick={onRowClick}
    />
  );
}
