"use client";

import React, { FC, useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Ban,
  Building2,
  CheckCircle,
  FileText,
  MessageSquare,
  Search,
  Users,
} from "lucide-react";
import { useAction } from "next-safe-action/hooks";
import { toast } from "sonner";

import { banUserAction } from "@/users/actions/banUserAction";
import { unbanUserAction } from "@/users/actions/unbanUserAction";
import { formatDate } from "@/lib/utils/date";
import { SectionHeading } from "@/components/ui/section-heading";
import { Stat } from "@/components/ui/stat";
import { useErrorHandling } from "@/lib/hooks/useErrorHandling";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AdminDataTable, type ColumnDef } from "@/components/AdminDataTable";
import { cn } from "@/lib/utils";

type AdminUser = {
  id: string;
  name: string | null;
  email: string | null;
  emailVerified: Date | null;
  image: string | null;
  isBanned: boolean;
  bannedAt: Date | null;
  banReason: string | null;
  bannedBy: {
    id: string;
    name: string | null;
    email: string | null;
  } | null;
  postCount: number;
  commentCount: number;
  institutionCount: number;
};

type AdminUsersPagination = {
  page: number;
  limit: number;
  totalCount: number;
  totalPages: number;
  hasNextPage: boolean;
  hasPrevPage: boolean;
};

const columnsKey = Symbol("adminUsersColumns");

export const AdminUsersView: FC = () => {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [showBannedOnly, setShowBannedOnly] = useState(false);
  const [pagination, setPagination] = useState<AdminUsersPagination | null>(
    null
  );
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [banReason, setBanReason] = useState("");
  const [showBanModal, setShowBanModal] = useState(false);

  const {
    error: banError,
    setError: setBanError,
    clearError: clearBanError,
  } = useErrorHandling({
    fallbackMessage: "Failed to ban user",
  });

  const {
    error: unbanError,
    setError: setUnbanError,
    clearError: clearUnbanError,
  } = useErrorHandling({
    fallbackMessage: "Failed to unban user",
  });

  const loadUsers = useCallback(async () => {
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        page: currentPage.toString(),
        limit: "20",
        ...(searchQuery && { search: searchQuery }),
        ...(showBannedOnly && { banned: "true" }),
      });
      const response = await fetch(`/api/admin/users?${params}`);
      if (!response.ok) {
        throw new Error("Failed to fetch users");
      }
      const data = await response.json();
      setUsers(data.users);
      setPagination(data.pagination);
    } catch (error) {
      console.error("Error loading users:", error);
      toast.error("Failed to load users.");
    } finally {
      setIsLoading(false);
    }
  }, [currentPage, searchQuery, showBannedOnly]);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const { execute: banUser, isExecuting: isBanning } = useAction(
    banUserAction,
    {
      onSuccess: () => {
        setShowBanModal(false);
        setSelectedUser(null);
        setBanReason("");
        clearBanError();
        loadUsers();
        toast.success("User banned successfully");
      },
      onError: (error) => {
        console.error("Error banning user:", error);
        setBanError(error);
        toast.error(error.message ?? "Failed to ban user");
      },
    }
  );

  const { execute: unbanUser, isExecuting: isUnbanning } = useAction(
    unbanUserAction,
    {
      onSuccess: () => {
        clearUnbanError();
        loadUsers();
        toast.success("User unbanned");
      },
      onError: (error) => {
        console.error("Error unbanning user:", error);
        setUnbanError(error);
        toast.error(error.message ?? "Failed to unban user");
      },
    }
  );

  const handleBanUser = (userEmail: string) => {
    if (!banReason.trim()) {
      toast.error("Please provide a reason for the ban.");
      return;
    }

    const formData = new FormData();
    formData.append("userEmail", userEmail);
    formData.append("reason", banReason);
    banUser(formData);
  };

  const handleUnbanUser = (userEmail: string) => {
    if (!confirm("Are you sure you want to unban this user?")) {
      return;
    }

    const formData = new FormData();
    formData.append("userEmail", userEmail);
    unbanUser(formData);
  };

  const openBanModal = (userId: string) => {
    setSelectedUser(userId);
    setBanReason("");
    setShowBanModal(true);
  };

  const filteredUsers = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();
    return users.filter((user) => {
      if (!normalizedQuery) {
        return true;
      }

      return [user.name ?? "", user.email ?? ""].some((value) =>
        value.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [searchQuery, users]);

  const columns = useMemo<ColumnDef<AdminUser, unknown>[]>(
    () => [
      {
        accessorKey: "name",
        header: "User",
        cell: ({ row }) => {
          const user = row.original;
          return (
            <div className="flex items-start gap-3">
              <div
                className={cn(
                  "flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full text-sm font-semibold",
                  user.isBanned
                    ? "bg-red-100 text-red-800"
                    : "bg-blue-100 text-blue-700"
                )}
              >
                {user.name
                  ? user.name[0].toUpperCase()
                  : (user.email?.[0]?.toUpperCase() ?? "?")}
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-foreground font-medium">
                    {user.name || "Unnamed User"}
                  </span>
                  {user.isBanned && (
                    <Badge className="inline-flex items-center gap-1 bg-red-100 text-xs font-medium text-red-800">
                      <Ban className="h-3 w-3" />
                      Banned
                    </Badge>
                  )}
                  {user.emailVerified && (
                    <Badge className="inline-flex items-center gap-1 bg-emerald-100 text-xs font-medium text-emerald-700">
                      <CheckCircle className="h-3 w-3" />
                      Email verified
                    </Badge>
                  )}
                </div>
                <div className="text-muted-foreground text-xs">
                  {user.email}
                </div>
              </div>
            </div>
          );
        },
        sortingFn: "text",
      },
      {
        accessorKey: "postCount",
        header: "Posts",
        cell: ({ row }) => (
          <span className="text-muted-foreground inline-flex items-center gap-1 text-sm">
            <FileText className="h-3 w-3" />
            {row.original.postCount}
          </span>
        ),
      },
      {
        accessorKey: "commentCount",
        header: "Comments",
        cell: ({ row }) => (
          <span className="text-muted-foreground inline-flex items-center gap-1 text-sm">
            <MessageSquare className="h-3 w-3" />
            {row.original.commentCount}
          </span>
        ),
      },
      {
        accessorKey: "institutionCount",
        header: "Institutions",
        cell: ({ row }) => (
          <span className="text-muted-foreground inline-flex items-center gap-1 text-sm">
            <Building2 className="h-3 w-3" />
            {row.original.institutionCount}
          </span>
        ),
      },
      {
        accessorKey: "bannedAt",
        header: "Status",
        cell: ({ row }) => {
          const user = row.original;
          if (!user.isBanned) {
            return (
              <span className="text-muted-foreground text-sm">Active</span>
            );
          }

          return (
            <div className="space-y-1 text-sm text-red-700">
              <div>Banned</div>
              {user.banReason && (
                <div className="text-xs">Reason: {user.banReason}</div>
              )}
              {user.bannedBy && (
                <div className="text-xs text-red-600">
                  By {user.bannedBy.name || user.bannedBy.email} on{" "}
                  {formatDate(user.bannedAt)}
                </div>
              )}
            </div>
          );
        },
      },
      {
        id: "actions",
        header: "Actions",
        enableSorting: false,
        cell: ({ row }) => {
          const user = row.original;
          return user.isBanned ? (
            <Button
              size="sm"
              variant="outline"
              onClick={() => user.email && handleUnbanUser(user.email)}
              disabled={isUnbanning}
            >
              {isUnbanning ? "Unbanning..." : "Unban"}
            </Button>
          ) : (
            <Button
              size="sm"
              variant="destructive"
              onClick={() => openBanModal(user.id)}
              disabled={isBanning}
            >
              Ban User
            </Button>
          );
        },
      },
    ],
    [handleUnbanUser, isBanning, isUnbanning]
  );

  const selectedUserData = useMemo(
    () => users.find((user) => user.id === selectedUser) ?? null,
    [selectedUser, users]
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1">
          <Search className="absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search users by name or email..."
            value={searchQuery}
            onChange={(event) => {
              setSearchQuery(event.target.value);
              setCurrentPage(1);
            }}
            className="border-border text-foreground focus:border-primary focus:ring-primary/20 w-full rounded-lg border py-2 pr-4 pl-10 text-sm focus:ring-2 focus:outline-none"
          />
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={showBannedOnly}
            onChange={(event) => {
              setShowBannedOnly(event.target.checked);
              setCurrentPage(1);
            }}
            className="border-border h-4 w-4 rounded text-red-600 focus:ring-red-500"
          />
          <span className="text-muted-foreground">Show banned only</span>
        </label>
      </div>

      {pagination && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3 md:gap-4">
          <Stat label="Total Users" value={pagination.totalCount} />
          <Stat
            label="Currently Banned"
            value={users.filter((user) => user.isBanned).length}
            helperText="Includes active bans"
          />
          <Stat
            label="Average Posts"
            value={
              users.length > 0
                ? Math.round(
                    users.reduce((sum, user) => sum + user.postCount, 0) /
                      users.length
                  )
                : 0
            }
            helperText="Per user"
          />
        </div>
      )}

      <AdminDataTable
        data={filteredUsers}
        columns={columns}
        isLoading={isLoading}
        emptyMessage={
          searchQuery
            ? "No users found matching your search."
            : "No users found."
        }
        onRowClick={(user) => setSelectedUser(user.id)}
        initialSorting={[{ id: "name", desc: false }]}
      />

      {pagination && pagination.totalPages > 1 && (
        <div className="mt-6 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="text-muted-foreground text-sm">
            Showing {(pagination.page - 1) * pagination.limit + 1}–
            {Math.min(
              pagination.page * pagination.limit,
              pagination.totalCount
            )}{" "}
            of
            {pagination.totalCount}
          </div>
          <div className="flex justify-center gap-2 md:justify-end">
            <Button
              variant="outline"
              size="sm"
              disabled={!pagination.hasPrevPage}
              onClick={() => setCurrentPage((page) => page - 1)}
            >
              Previous
            </Button>
            <span className="text-muted-foreground flex items-center px-3 py-2 text-sm">
              {pagination.page}/{pagination.totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={!pagination.hasNextPage}
              onClick={() => setCurrentPage((page) => page + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      )}

      {showBanModal && selectedUserData && (
        <div className="bg-opacity-50 fixed inset-0 z-50 flex items-center justify-center bg-black p-4">
          <div className="w-full max-w-md rounded-lg bg-white p-4 shadow-xl md:p-6">
            <h3 className="mb-3 text-base font-semibold text-gray-900 md:mb-4 md:text-lg">
              Ban User: {selectedUserData.name || selectedUserData.email}
            </h3>

            <div className="mb-4">
              <label className="mb-2 block text-sm font-medium text-gray-700">
                Reason for ban *
              </label>
              <textarea
                value={banReason}
                onChange={(event) => setBanReason(event.target.value)}
                placeholder="Please provide a detailed reason for banning this user..."
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-red-500 focus:ring-1 focus:ring-red-500 focus:outline-none"
                rows={3}
                maxLength={500}
              />
              <div className="mt-1 text-xs text-gray-500">
                {banReason.length}/500 characters
              </div>
            </div>

            <div className="flex justify-end gap-2 md:gap-3">
              <Button
                variant="outline"
                onClick={() => {
                  setShowBanModal(false);
                  setSelectedUser(null);
                  setBanReason("");
                  clearBanError();
                }}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={() =>
                  selectedUserData.email &&
                  handleBanUser(selectedUserData.email)
                }
                disabled={!banReason.trim() || isBanning}
              >
                {isBanning ? "Banning..." : "Ban User"}
              </Button>
            </div>
            {banError && (
              <p className="mt-3 text-sm text-red-600">{banError}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
