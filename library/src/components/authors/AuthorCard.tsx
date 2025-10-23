"use client";

import React from "react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Stat } from "@/components/ui/stat";
import { cn } from "@/lib/utils";
import { getInitialsFromName } from "@/lib/utils/text";
import { AuthorDTO } from "@/posts/data/authors-client";

interface AuthorCardProps {
  author: AuthorDTO;
  onClick?: (author: AuthorDTO) => void;
  onToggleFollow?: (author: AuthorDTO) => void;
  onExpertiseClick?: (expertise: string) => void;
}

export function AuthorCard({
  author,
  onClick,
  onToggleFollow,
  onExpertiseClick,
}: AuthorCardProps) {
  const hIndex = author.hIndex ?? 0;
  const totalCitations = author.totalCitations ?? 0;

  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={() => onClick?.(author)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick?.(author);
        }
      }}
      className="group border-muted focus-visible:ring-primary flex h-full cursor-pointer flex-col shadow-sm transition-shadow hover:shadow-md focus-visible:ring-2 focus-visible:outline-none"
    >
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div
            className={cn(
              "flex h-12 w-12 items-center justify-center rounded-xl text-sm font-semibold",
              author.claimed
                ? "bg-blue-500 text-white"
                : "bg-primary/10 text-primary"
            )}
          >
            {author.avatar || getInitialsFromName(author.name, 2)}
          </div>
          <div className="min-w-0 space-y-1">
            <CardTitle className="text-foreground text-lg break-words">
              {author.name}
            </CardTitle>
            {author.institution && (
              <p className="text-muted-foreground text-sm break-words">
                {author.institution}
              </p>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          {author.claimed && onToggleFollow && (
            <Button
              variant={author.isFollowing ? "secondary" : "default"}
              size="sm"
              onClick={(event) => {
                event.stopPropagation();
                onToggleFollow(author);
              }}
            >
              {author.isFollowing ? "Following" : "Follow"}
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-3">
          <Stat label="h-index" value={hIndex} />
          <Stat label="Papers" value={author.paperCount} />
          <Stat label="Citations" value={totalCitations.toLocaleString()} />
        </div>

        {/* Expertise Tags */}
        {author.expertise.length > 0 && (
          <div className="space-y-2">
            <p className="text-muted-foreground text-xs font-semibold tracking-wide uppercase">
              Expertise
            </p>
            <div className="flex flex-wrap gap-2">
              {author.expertise.slice(0, 5).map((exp) => (
                <Badge
                  key={exp}
                  variant="outline"
                  className="hover:bg-accent cursor-pointer transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    onExpertiseClick?.(exp);
                  }}
                >
                  {exp}
                </Badge>
              ))}
              {author.expertise.length > 5 && (
                <Badge variant="outline" className="text-muted-foreground">
                  +{author.expertise.length - 5}
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>

      <CardFooter className="text-muted-foreground mt-auto flex items-center justify-between text-xs">
        <span>{author.claimed ? "Claimed profile" : "Unclaimed"}</span>
      </CardFooter>
    </Card>
  );
}

AuthorCard.displayName = "AuthorCard";
