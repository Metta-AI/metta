"use client";

import React from "react";
import { Globe, Lock, Building } from "lucide-react";

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
import { formatDate } from "@/lib/utils/date";
import { GroupDTO } from "@/posts/data/groups";

type GroupMode = "directory" | "member";

interface GroupCardProps {
  group: GroupDTO;
  mode: GroupMode;
  isJoining?: boolean;
  onClick?: (group: GroupDTO) => void;
  onJoin?: (group: GroupDTO) => void;
  onManage?: (group: GroupDTO) => void;
}

const roleBadgeVariants: Record<string, { label: string; className: string }> =
  {
    admin: {
      label: "Admin",
      className: "bg-blue-100 text-blue-700",
    },
    member: {
      label: "Member",
      className: "bg-slate-100 text-slate-700",
    },
  };

export function GroupCard({
  group,
  mode,
  isJoining = false,
  onClick,
  onJoin,
  onManage,
}: GroupCardProps) {
  const canJoin = mode === "directory" && !group.currentUserRole;
  const isMember = mode === "member" || !!group.currentUserRole;

  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={() => onClick?.(group)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick?.(group);
        }
      }}
      className={cn(
        "group border-muted focus-visible:ring-primary flex h-full cursor-pointer flex-col shadow-sm transition-shadow hover:shadow-md focus-visible:ring-2 focus-visible:outline-none",
        isMember && "border-blue-200 bg-blue-50"
      )}
    >
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-600 text-sm font-semibold text-white">
            {getInitialsFromName(group.name, 2)}
          </div>
          <div className="min-w-0 space-y-1">
            <CardTitle className="text-foreground text-lg break-words">
              {group.name}
            </CardTitle>
            <CardDescription className="text-muted-foreground flex items-center gap-1 text-sm">
              {group.isPublic ? (
                <>
                  <Globe className="h-3 w-3" />
                  Public
                </>
              ) : (
                <>
                  <Lock className="h-3 w-3" />
                  Private
                </>
              )}
            </CardDescription>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          {group.currentUserRole &&
            roleBadgeVariants[group.currentUserRole]?.label && (
              <Badge
                className={cn(
                  "text-xs font-semibold",
                  roleBadgeVariants[group.currentUserRole].className
                )}
              >
                {roleBadgeVariants[group.currentUserRole].label}
              </Badge>
            )}
          {onManage && group.currentUserRole === "admin" && (
            <Button
              variant="outline"
              size="sm"
              onClick={(event) => {
                event.stopPropagation();
                onManage(group);
              }}
            >
              Manage
            </Button>
          )}
          {canJoin && onJoin && group.isPublic && (
            <Button
              size="sm"
              onClick={(event) => {
                event.stopPropagation();
                onJoin(group);
              }}
              disabled={isJoining}
            >
              {isJoining ? "Joining..." : "Join"}
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Description */}
        {group.description && (
          <p className="text-muted-foreground line-clamp-2 text-sm">
            {group.description}
          </p>
        )}

        {/* Stats */}
        <div className="grid grid-cols-1 gap-3">
          <Stat label="Members" value={group.memberCount} />
        </div>

        {/* Institution */}
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <Building className="h-4 w-4" />
          <span className="truncate">{group.institution.name}</span>
        </div>
      </CardContent>

      <CardFooter className="text-muted-foreground mt-auto flex items-center justify-between text-xs">
        <span>Created {formatDate(group.createdAt)}</span>
      </CardFooter>
    </Card>
  );
}

GroupCard.displayName = "GroupCard";
