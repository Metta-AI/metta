"use client";

import React, { FC, useState, useMemo } from "react";
import { Plus, Search } from "lucide-react";

import { GroupCreateForm } from "./GroupCreateForm";
import { GroupManagementModal } from "./GroupManagementModal";
import { GroupCard } from "@/components/groups/GroupCard";
import { GroupDTO } from "@/posts/data/groups";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useFilterSort } from "@/lib/hooks/useFilterSort";
import { SortControls } from "@/components/ui/sort-controls";
import { useJoinGroup } from "@/hooks/mutations/useJoinGroup";

interface GroupsViewProps {
  userGroups: GroupDTO[];
  allGroups: GroupDTO[];
  userInstitutions: Array<{
    id: string;
    name: string;
  }>;
}

export const GroupsView: FC<GroupsViewProps> = ({
  userGroups,
  allGroups,
  userInstitutions,
}) => {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<GroupDTO | null>(null);

  const { mutate: joinGroup, isPending: isJoining } = useJoinGroup();

  const {
    searchQuery,
    setSearchQuery,
    sortBy,
    setSortBy,
    sortDirection,
    setSortDirection,
    filteredAndSortedItems: filteredAndSortedGroups,
  } = useFilterSort<GroupDTO>(allGroups, {
    getSearchableValues: (group) => [
      group.name,
      group.description ?? "",
      group.institution.name,
    ],
    sorters: {
      name: (group) => group.name.toLowerCase(),
      memberCount: (group) => group.memberCount,
      createdAt: (group) => new Date(group.createdAt).getTime(),
    },
    initialSortKey: "memberCount",
    initialSortDirection: "desc",
  });

  const handleJoinGroup = (group: GroupDTO) => {
    joinGroup({ groupId: group.id });
  };

  // Helper to determine if user can join a group
  const canJoinGroup = (group: GroupDTO) => {
    // User must not already be a member
    if (group.currentUserRole) return false;

    // Group must be public
    if (!group.isPublic) return false;

    // User must be a member of the same institution
    return userInstitutions.some((inst) => inst.id === group.institution.id);
  };

  // Helper to get group with full member data
  const getGroupWithMembers = (group: GroupDTO) => {
    // For user groups (where user is a member), get the version with full member data
    const userGroup = userGroups.find((ug) => ug.id === group.id);
    return userGroup || group;
  };

  const sortOptions = useMemo(
    () => [
      { key: "memberCount", label: "Members" },
      { key: "name", label: "Name" },
      { key: "createdAt", label: "Created" },
    ],
    []
  );

  const userGroupsSet = useMemo(
    () => new Set(userGroups.map((g) => g.id)),
    [userGroups]
  );

  return (
    <div className="flex h-full w-full flex-col">
      {/* Header Section */}
      <div className="border-b border-gray-200 bg-white px-4 py-4 md:px-6 md:py-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">Groups</h1>
          <span className="text-sm text-gray-500">
            Showing {filteredAndSortedGroups.length} of {allGroups.length}{" "}
            groups
          </span>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
        <div className="mx-auto w-full max-w-7xl space-y-6">
          {/* Stats and Actions Bar */}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{userGroups.length} joined</Badge>
              <Badge variant="secondary">{allGroups.length} total</Badge>
            </div>
            <Button onClick={() => setShowCreateForm(true)} size="sm">
              <Plus className="h-4 w-4" />
              Create Group
            </Button>
          </div>

          {/* My Groups Section */}
          {userGroups.length > 0 && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-gray-900">
                My Groups ({userGroups.length})
              </h2>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
                {userGroups.map((group) => (
                  <GroupCard
                    key={group.id}
                    group={group}
                    mode="member"
                    onClick={() => setSelectedGroup(group)}
                    onManage={(g) => setSelectedGroup(getGroupWithMembers(g))}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Search Bar */}
          <div className="relative w-full">
            <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2" />
            <Input
              placeholder="Search groups"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="pl-10"
            />
          </div>

          {/* Sort Controls */}
          <SortControls
            sortOptions={sortOptions}
            sortBy={sortBy}
            sortDirection={sortDirection}
            onSortChange={(key) => {
              setSortBy(key);
              setSortDirection(key === "name" ? "asc" : "desc");
            }}
            onDirectionToggle={() =>
              setSortDirection(sortDirection === "asc" ? "desc" : "asc")
            }
          />

          {/* All Groups Section */}
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Discover Groups
            </h2>

            {/* Groups Grid */}
            {filteredAndSortedGroups.length > 0 ? (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
                {filteredAndSortedGroups.map((group) => {
                  const mode = userGroupsSet.has(group.id)
                    ? "member"
                    : "directory";

                  return (
                    <GroupCard
                      key={group.id}
                      group={group}
                      mode={mode}
                      isJoining={isJoining}
                      onClick={(g) => {
                        if (g.currentUserRole) {
                          setSelectedGroup(getGroupWithMembers(g));
                        }
                      }}
                      onJoin={canJoinGroup(group) ? handleJoinGroup : undefined}
                      onManage={
                        group.currentUserRole === "admin"
                          ? (g) => setSelectedGroup(getGroupWithMembers(g))
                          : undefined
                      }
                    />
                  );
                })}
              </div>
            ) : (
              <div className="border-border bg-card text-muted-foreground rounded-xl border p-8 text-center">
                No groups found.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Modals */}
      <GroupCreateForm
        isOpen={showCreateForm}
        onClose={() => setShowCreateForm(false)}
        userInstitutions={userInstitutions}
      />

      {selectedGroup && (
        <GroupManagementModal
          isOpen={!!selectedGroup}
          onClose={() => setSelectedGroup(null)}
          group={selectedGroup}
          currentUserRole={selectedGroup.currentUserRole}
        />
      )}
    </div>
  );
};
