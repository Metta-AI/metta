"use client";

import React, { FC, useState, useMemo } from "react";
import {
  Plus,
  Search,
  ChevronDown,
  Users,
  Globe,
  Lock,
  Calendar,
} from "lucide-react";

import { GroupCreateForm } from "./GroupCreateForm";
import { GroupManagementModal } from "./GroupManagementModal";
import { GroupDTO } from "@/posts/data/groups";

interface GroupsViewProps {
  userGroups: GroupDTO[];
  allGroups: GroupDTO[];
}

export const GroupsView: FC<GroupsViewProps> = ({ userGroups, allGroups }) => {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<GroupDTO | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState("memberCount");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");

  // Helper to get group with full member data
  const getGroupWithMembers = (group: GroupDTO) => {
    // For user groups (where user is a member), get the version with full member data
    const userGroup = userGroups.find((ug) => ug.id === group.id);
    return userGroup || group;
  };

  // Filter and sort groups based on current state
  const filteredAndSortedGroups = useMemo(() => {
    let filtered = allGroups;

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (group) =>
          group.name.toLowerCase().includes(query) ||
          (group.description?.toLowerCase().includes(query) ?? false)
      );
    }

    // Apply sorting
    const sortedFiltered = [...filtered].sort((a, b) => {
      let valueA: any, valueB: any;

      switch (sortBy) {
        case "name":
          valueA = a.name.toLowerCase();
          valueB = b.name.toLowerCase();
          break;
        case "memberCount":
          valueA = a.memberCount;
          valueB = b.memberCount;
          break;
        case "createdAt":
          valueA = new Date(a.createdAt).getTime();
          valueB = new Date(b.createdAt).getTime();
          break;
        default:
          valueA = a.memberCount;
          valueB = b.memberCount;
      }

      if (sortDirection === "asc") {
        return valueA > valueB ? 1 : valueA < valueB ? -1 : 0;
      } else {
        return valueA < valueB ? 1 : valueA > valueB ? -1 : 0;
      }
    });

    return sortedFiltered;
  }, [allGroups, searchQuery, sortBy, sortDirection]);

  const getGroupInitials = (name: string) => {
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const handleSortClick = (key: string) => {
    if (sortBy === key) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortBy(key);
      setSortDirection(key === "name" ? "asc" : "desc");
    }
  };

  const formatDate = (date: Date | string | null) => {
    if (!date) return "Unknown";
    const dateObj = typeof date === "string" ? new Date(date) : date;
    if (isNaN(dateObj.getTime())) return "Unknown";

    const now = new Date();
    const diffInDays = Math.floor(
      (now.getTime() - dateObj.getTime()) / (1000 * 60 * 60 * 24)
    );

    if (diffInDays === 0) return "Today";
    if (diffInDays === 1) return "Yesterday";
    if (diffInDays < 7) return `${diffInDays} days ago`;
    if (diffInDays < 30) return `${Math.floor(diffInDays / 7)} weeks ago`;
    return `${Math.floor(diffInDays / 30)} months ago`;
  };

  const sortOptions = [
    { key: "memberCount", label: "Members" },
    { key: "name", label: "Name" },
    { key: "createdAt", label: "Created" },
  ];

  return (
    <div className="p-4">
      {/* Header with Create Button */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Groups</h1>
          <p className="text-gray-600">Interest groups and teams</p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700"
        >
          <Plus className="h-4 w-4" />
          Create Group
        </button>
      </div>

      {/* My Groups Section */}
      {userGroups.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-4 text-lg font-semibold text-gray-900">
            My Groups ({userGroups.length})
          </h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {userGroups.map((group) => (
              <div
                key={group.id}
                className="cursor-pointer rounded-lg border border-blue-200 bg-blue-50 p-4 transition-all hover:border-blue-300 hover:shadow-md"
                onClick={() => {
                  // For user groups, we always have member data
                  setSelectedGroup(group);
                }}
              >
                {/* Group Header */}
                <div className="mb-3 flex items-start gap-3">
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-blue-600 text-sm font-semibold text-white">
                    {getGroupInitials(group.name)}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <h3 className="truncate text-base font-semibold text-gray-900">
                        {group.name}
                      </h3>
                      {group.currentUserRole === "admin" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedGroup(getGroupWithMembers(group));
                          }}
                          className="ml-2 rounded-md p-1 text-gray-400 transition-colors hover:bg-blue-100 hover:text-gray-600"
                          title="Manage group"
                        >
                          <Users className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      {group.isPublic ? (
                        <Globe className="h-3 w-3" />
                      ) : (
                        <Lock className="h-3 w-3" />
                      )}
                      <span>{group.memberCount} members</span>
                    </div>
                  </div>
                </div>

                {group.description && (
                  <p className="mb-3 line-clamp-2 text-sm text-gray-600">
                    {group.description}
                  </p>
                )}

                {/* User Info */}
                <div className="flex items-center justify-between text-xs">
                  <span className="rounded bg-blue-100 px-2 py-1 font-medium text-blue-700">
                    {group.currentUserRole}
                  </span>
                  <span className="text-gray-500">
                    Created {formatDate(group.createdAt)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* All Groups Section */}
      <div className="mb-6">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">
          Discover Groups
        </h2>

        {/* Filter and Sort Controls */}
        <div className="mb-6 space-y-4">
          {/* Search Input */}
          <div className="relative">
            <Search className="absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2 transform text-gray-400" />
            <input
              type="text"
              placeholder="Search groups..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded-lg border border-gray-300 py-3 pr-4 pl-10 text-gray-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
            />
          </div>

          {/* Sort Options */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-gray-700">Sort by:</span>
            {sortOptions.map((option) => (
              <button
                key={option.key}
                onClick={() => handleSortClick(option.key)}
                className={`flex items-center gap-1 rounded-full px-3 py-1 text-sm transition-colors ${
                  sortBy === option.key
                    ? "bg-blue-100 text-blue-700"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {option.label}
                {sortBy === option.key && (
                  <ChevronDown
                    className={`h-4 w-4 transition-transform ${
                      sortDirection === "asc" ? "rotate-180" : ""
                    }`}
                  />
                )}
              </button>
            ))}
          </div>

          {/* Results Count */}
          <div className="text-sm text-gray-500">
            Showing {filteredAndSortedGroups.length} of {allGroups.length}{" "}
            groups
          </div>
        </div>

        {/* Groups Grid */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredAndSortedGroups.map((group) => {
            const isUserGroup = userGroups.some((ug) => ug.id === group.id);

            return (
              <div
                key={group.id}
                className={`cursor-pointer rounded-lg border p-4 transition-all hover:shadow-md ${
                  isUserGroup
                    ? "border-blue-200 bg-blue-50 hover:border-blue-300"
                    : "border-gray-200 bg-white hover:border-gray-300"
                }`}
                onClick={() => {
                  if (group.currentUserRole) {
                    setSelectedGroup(getGroupWithMembers(group));
                  }
                }}
              >
                {/* Group Header */}
                <div className="mb-3 flex items-start gap-3">
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-blue-600 text-sm font-semibold text-white">
                    {getGroupInitials(group.name)}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <h3 className="truncate text-base font-semibold text-gray-900">
                        {group.name}
                      </h3>
                      {group.currentUserRole === "admin" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedGroup(getGroupWithMembers(group));
                          }}
                          className="ml-2 rounded-md p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
                          title="Manage group"
                        >
                          <Users className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      {group.isPublic ? (
                        <Globe className="h-3 w-3" />
                      ) : (
                        <Lock className="h-3 w-3" />
                      )}
                      <span>{group.memberCount} members</span>
                    </div>
                  </div>
                </div>

                {group.description && (
                  <p className="mb-3 line-clamp-2 text-sm text-gray-600">
                    {group.description}
                  </p>
                )}

                {/* Footer */}
                <div className="flex items-center justify-between text-xs">
                  {group.currentUserRole ? (
                    <span className="rounded bg-blue-100 px-2 py-1 font-medium text-blue-700">
                      {group.currentUserRole}
                    </span>
                  ) : (
                    <span className="text-gray-500">
                      {group.isPublic ? "Public" : "Private"}
                    </span>
                  )}
                  <span className="text-gray-500">
                    Created {formatDate(group.createdAt)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Empty State */}
        {filteredAndSortedGroups.length === 0 && (
          <div className="py-8 text-center">
            <Users className="mx-auto mb-4 h-12 w-12 text-gray-400" />
            <p className="text-gray-500">
              {searchQuery
                ? "No groups found matching your search."
                : "No groups found."}
            </p>
          </div>
        )}
      </div>

      {/* Modals */}
      <GroupCreateForm
        isOpen={showCreateForm}
        onClose={() => setShowCreateForm(false)}
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
