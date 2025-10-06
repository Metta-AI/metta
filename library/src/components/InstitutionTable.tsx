"use client";

import React, { useMemo } from "react";
import { createColumnHelper } from "@tanstack/react-table";

import { AdminDataTable } from "@/components/AdminDataTable";
import type { InstitutionCardData } from "@/components/institutions/InstitutionCard";
import { getInitialsFromName } from "@/lib/utils/text";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface InstitutionTableProps<TInstitution extends InstitutionCardData> {
  data: TInstitution[];
  isLoading?: boolean;
  onSelect?: (institution: TInstitution) => void;
}

const columnHelper = createColumnHelper<InstitutionCardData>();

export function InstitutionTable<TInstitution extends InstitutionCardData>({
  data,
  isLoading = false,
  onSelect,
}: InstitutionTableProps<TInstitution>) {
  const columns = useMemo(() => {
    return [
      columnHelper.accessor("name", {
        header: "Institution",
        cell: (info) => {
          const institution = info.row.original;
          return (
            <div className="flex items-center gap-3">
              <div className="bg-primary/10 text-primary flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold">
                {getInitialsFromName(institution.name)}
              </div>
              <div>
                <p className="text-foreground font-medium">
                  {institution.name}
                </p>
                <p className="text-muted-foreground text-xs">
                  {institution.domain ?? "No domain"}
                </p>
              </div>
            </div>
          );
        },
      }),
      columnHelper.accessor("paperCount", {
        header: "Papers",
      }),
      columnHelper.accessor((row) => row.avgStars.toFixed(1), {
        id: "avgStars",
        header: "Avg Stars",
      }),
      columnHelper.accessor("authorCount", {
        header: "Authors",
      }),
      columnHelper.accessor("memberCount", {
        header: "Members",
      }),
      columnHelper.display({
        id: "status",
        header: "Status",
        cell: ({ row }) => {
          const institution = row.original;
          return (
            <div className="flex gap-2">
              {institution.isVerified && (
                <Badge className="bg-emerald-100 text-emerald-700">
                  Verified
                </Badge>
              )}
              {institution.requiresApproval && (
                <Badge className="bg-amber-100 text-amber-800">
                  Approval required
                </Badge>
              )}
            </div>
          );
        },
      }),
      columnHelper.display({
        id: "recentActivity",
        header: "Recent Activity",
        cell: ({ row }) => (
          <span className="text-muted-foreground text-xs">
            {row.original.recentActivity
              ? new Date(row.original.recentActivity).toLocaleDateString()
              : "Unknown"}
          </span>
        ),
      }),
    ];
  }, []);

  return (
    <AdminDataTable
      data={data}
      columns={columns}
      isLoading={isLoading}
      onRowClick={onSelect}
      emptyMessage="No institutions found."
      initialSorting={[{ id: "paperCount", desc: true }]}
    />
  );
}
