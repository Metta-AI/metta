"use client";

import React from "react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Stat } from "@/components/ui/stat";
import { cn } from "@/lib/utils";
import { getInitialsFromName } from "@/lib/utils/text";

type InstitutionsDirectoryMode = "directory" | "member" | "admin";

export interface InstitutionCardData {
  id: string;
  name: string;
  domain?: string | null;
  memberCount?: number;
  paperCount: number;
  authorCount?: number;
  totalStars?: number;
  avgStars: number;
  topCategories?: string[];
  isVerified?: boolean;
  requiresApproval?: boolean;
  recentActivity?: string | Date | null;
  currentUserRole?: string | null;
  currentUserStatus?: string | null;
  members?: Array<{ id: string }>;
}

interface InstitutionCardProps<TInstitution extends InstitutionCardData> {
  institution: TInstitution;
  mode: InstitutionsDirectoryMode;
  isJoining?: boolean;
  onClick?: (institution: TInstitution) => void;
  onJoin?: (institution: TInstitution) => void;
  onManage?: (institution: TInstitution) => void;
}

const statusBadgeVariants: Record<
  string,
  { label: string; className: string }
> = {
  admin: {
    label: "Admin",
    className: "bg-blue-100 text-blue-700",
  },
  member: {
    label: "Member",
    className: "bg-slate-100 text-slate-700",
  },
};

export function InstitutionCard<TInstitution extends InstitutionCardData>({
  institution,
  mode,
  isJoining = false,
  onClick,
  onJoin,
  onManage,
}: InstitutionCardProps<TInstitution>) {
  const canJoin =
    mode !== "admin" &&
    !institution.currentUserRole &&
    !institution.currentUserStatus;

  const topCategories = institution.topCategories ?? [];
  const memberCount = institution.memberCount ?? 0;
  const authorCount = institution.authorCount ?? 0;
  const totalStars = institution.totalStars ?? 0;
  const avgStars = Number.isFinite(institution.avgStars)
    ? institution.avgStars
    : 0;
  const recentActivityLabel = institution.recentActivity
    ? new Date(institution.recentActivity).toLocaleDateString()
    : "Unknown";

  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={() => onClick?.(institution)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick?.(institution);
        }
      }}
      className="group border-muted focus-visible:ring-primary flex h-full flex-col shadow-sm transition-shadow hover:shadow-md focus-visible:ring-2 focus-visible:outline-none cursor-pointer"
    >
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="bg-primary/10 text-primary flex h-12 w-12 items-center justify-center rounded-xl text-sm font-semibold">
            {getInitialsFromName(institution.name)}
          </div>
          <div className="space-y-1">
            <CardTitle className="text-foreground text-lg">
              {institution.name}
            </CardTitle>
            <CardDescription className="text-muted-foreground text-sm">
              {institution.domain ?? "No domain"}
            </CardDescription>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          {institution.isVerified && (
            <Badge
              variant="secondary"
              className="bg-emerald-100 text-emerald-700"
            >
              Verified
            </Badge>
          )}
          {institution.requiresApproval && (
            <Badge variant="secondary" className="bg-amber-100 text-amber-800">
              Approval required
            </Badge>
          )}
          {institution.currentUserRole &&
            statusBadgeVariants[institution.currentUserRole]?.label && (
              <Badge
                className={cn(
                  "text-xs font-semibold",
                  statusBadgeVariants[institution.currentUserRole].className
                )}
              >
                {statusBadgeVariants[institution.currentUserRole].label}
              </Badge>
            )}
          {institution.currentUserStatus === "PENDING" && (
            <Badge className="bg-amber-50 text-amber-700">
              Pending approval
            </Badge>
          )}
          {institution.currentUserStatus === "REJECTED" && (
            <Badge className="bg-rose-50 text-rose-700">Request rejected</Badge>
          )}
          {onManage && institution.currentUserRole === "admin" && (
            <Button
              variant="outline"
              size="sm"
              onClick={(event) => {
                event.stopPropagation();
                onManage(institution);
              }}
            >
              Manage
            </Button>
          )}
          {canJoin && onJoin && (
            <Button
              size="sm"
              onClick={(event) => {
                event.stopPropagation();
                onJoin(institution);
              }}
              disabled={isJoining}
            >
              {isJoining ? "Joining..." : "Join"}
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Papers" value={institution.paperCount} />
          <Stat
            label="Avg stars"
            value={avgStars.toFixed(1)}
            helperText={`${totalStars} total`}
          />
          <Stat label="Authors" value={authorCount} />
          <Stat label="Members" value={memberCount} />
        </div>

        {topCategories.length > 0 && (
          <div className="space-y-2">
            <p className="text-muted-foreground text-xs font-semibold tracking-wide uppercase">
              Top areas
            </p>
            <div className="flex flex-wrap gap-2">
              {topCategories.slice(0, 5).map((category) => (
                <Badge key={category} variant="outline">
                  {category}
                </Badge>
              ))}
              {topCategories.length > 5 && (
                <Badge variant="outline" className="text-muted-foreground">
                  +{topCategories.length - 5}
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>

      <CardFooter className="text-muted-foreground mt-auto flex items-center justify-between text-xs">
        <span>Last activity: {recentActivityLabel}</span>
        {mode === "admin" && institution.members && (
          <span>{institution.members.length} admins</span>
        )}
      </CardFooter>
    </Card>
  );
}

InstitutionCard.displayName = "InstitutionCard";
