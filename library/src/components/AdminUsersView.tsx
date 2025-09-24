"use client";

import React, { FC, useState, useEffect } from "react";
import {
  Users,
  Shield,
  Search,
  Ban,
  CheckCircle,
  UserX,
  MessageSquare,
  FileText,
  Building2,
  AlertTriangle,
} from "lucide-react";
import { useAction } from "next-safe-action/hooks";

import { banUserAction } from "@/users/actions/banUserAction";
import { unbanUserAction } from "@/users/actions/unbanUserAction";

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

export const AdminUsersView: FC = () => {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [showBannedOnly, setShowBannedOnly] = useState(false);
  const [pagination, setPagination] = useState<AdminUsersPagination | null>(
    null
  );
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [banReason, setBanReason] = useState("");
  const [showBanModal, setShowBanModal] = useState(false);

  // Load users
  useEffect(() => {
    loadUsers();
  }, [currentPage, showBannedOnly, searchQuery]);

  const loadUsers = async () => {
    setLoading(true);
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
      alert("Failed to load users.");
    } finally {
      setLoading(false);
    }
  };

  // Ban user action
  const { execute: banUser, isExecuting: isBanning } = useAction(
    banUserAction,
    {
      onSuccess: () => {
        setShowBanModal(false);
        setSelectedUser(null);
        setBanReason("");
        loadUsers(); // Reload data
      },
      onError: (error) => {
        console.error("Error banning user:", error);
        alert(error.error?.serverError || "Failed to ban user");
      },
    }
  );

  // Unban user action
  const { execute: unbanUser, isExecuting: isUnbanning } = useAction(
    unbanUserAction,
    {
      onSuccess: () => {
        loadUsers(); // Reload data
      },
      onError: (error) => {
        console.error("Error unbanning user:", error);
        alert(error.error?.serverError || "Failed to unban user");
      },
    }
  );

  const handleBanUser = (userEmail: string) => {
    if (!banReason.trim()) {
      alert("Please provide a reason for the ban.");
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

  const formatDate = (date: Date | string | null) => {
    if (!date) return "Never";
    const d = new Date(date);
    return d.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (loading && users.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">Loading users...</div>
    );
  }

  const selectedUserData = users.find((user) => user.id === selectedUser);

  return (
    <div className="p-8">
      {/* Header with filters */}
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search users by name or email..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full rounded-lg border border-gray-300 py-3 pr-4 pl-10 text-gray-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
          />
        </div>

        {/* Filter toggle */}
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={showBannedOnly}
            onChange={(e) => {
              setShowBannedOnly(e.target.checked);
              setCurrentPage(1);
            }}
            className="rounded border-gray-300 text-red-600 focus:ring-red-500"
          />
          <span className="text-gray-700">Show banned users only</span>
        </label>
      </div>

      {/* Stats */}
      {pagination && (
        <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-3">
              <Users className="h-8 w-8 text-blue-600" />
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {pagination.totalCount}
                </div>
                <div className="text-sm text-gray-600">
                  {showBannedOnly ? "Banned Users" : "Total Users"}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Users List */}
      <div className="space-y-4">
        {users.length === 0 && (
          <div className="py-8 text-center">
            <UserX className="mx-auto mb-4 h-12 w-12 text-gray-400" />
            <p className="text-gray-600">
              {searchQuery
                ? "No users found matching your search."
                : showBannedOnly
                  ? "No banned users found."
                  : "No users found."}
            </p>
          </div>
        )}

        {users.map((user) => (
          <div
            key={user.id}
            className={`rounded-lg border bg-white p-6 shadow-sm ${
              user.isBanned ? "border-red-200 bg-red-50" : "border-gray-200"
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4">
                {/* Avatar */}
                <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
                  {user.name
                    ? user.name[0].toUpperCase()
                    : user.email?.[0].toUpperCase() || "?"}
                </div>

                {/* User Info */}
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-lg font-semibold text-gray-900">
                      {user.name || "Unnamed User"}
                    </h3>
                    {user.isBanned && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-1 text-xs font-medium text-red-800">
                        <Ban className="h-3 w-3" />
                        Banned
                      </span>
                    )}
                    {user.emailVerified && (
                      <CheckCircle
                        className="h-4 w-4 text-green-600"
                        title="Email verified"
                      />
                    )}
                  </div>

                  <p className="text-sm text-gray-600">{user.email}</p>

                  {/* Stats */}
                  <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
                    <span className="flex items-center gap-1">
                      <FileText className="h-3 w-3" />
                      {user.postCount} posts
                    </span>
                    <span className="flex items-center gap-1">
                      <MessageSquare className="h-3 w-3" />
                      {user.commentCount} comments
                    </span>
                    <span className="flex items-center gap-1">
                      <Building2 className="h-3 w-3" />
                      {user.institutionCount} institutions
                    </span>
                  </div>

                  {/* Ban info */}
                  {user.isBanned && user.banReason && (
                    <div className="mt-3 rounded-md bg-red-100 p-3">
                      <div className="flex items-start gap-2">
                        <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-600" />
                        <div className="text-sm">
                          <div className="font-medium text-red-800">
                            Ban Reason:
                          </div>
                          <div className="text-red-700">{user.banReason}</div>
                          {user.bannedBy && (
                            <div className="mt-1 text-xs text-red-600">
                              Banned by{" "}
                              {user.bannedBy.name || user.bannedBy.email} on{" "}
                              {formatDate(user.bannedAt)}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                {user.isBanned ? (
                  <button
                    onClick={() => handleUnbanUser(user.email!)}
                    disabled={isUnbanning}
                    className="flex items-center gap-1 rounded-md bg-green-600 px-3 py-1.5 text-sm text-white transition-colors hover:bg-green-700 disabled:opacity-50"
                  >
                    <CheckCircle className="h-4 w-4" />
                    {isUnbanning ? "Unbanning..." : "Unban"}
                  </button>
                ) : (
                  <button
                    onClick={() => openBanModal(user.id)}
                    disabled={isBanning}
                    className="flex items-center gap-1 rounded-md bg-red-600 px-3 py-1.5 text-sm text-white transition-colors hover:bg-red-700 disabled:opacity-50"
                  >
                    <Ban className="h-4 w-4" />
                    Ban User
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      {pagination && pagination.totalPages > 1 && (
        <div className="mt-8 flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {(pagination.page - 1) * pagination.limit + 1} to{" "}
            {Math.min(
              pagination.page * pagination.limit,
              pagination.totalCount
            )}{" "}
            of {pagination.totalCount} users
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentPage(pagination.page - 1)}
              disabled={!pagination.hasPrevPage}
              className="rounded-md border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
            >
              Previous
            </button>
            <span className="flex items-center px-3 py-2 text-sm text-gray-700">
              Page {pagination.page} of {pagination.totalPages}
            </span>
            <button
              onClick={() => setCurrentPage(pagination.page + 1)}
              disabled={!pagination.hasNextPage}
              className="rounded-md border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Ban Modal */}
      {showBanModal && selectedUserData && (
        <div className="bg-opacity-50 fixed inset-0 z-50 flex items-center justify-center bg-black">
          <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
            <h3 className="mb-4 text-lg font-semibold text-gray-900">
              Ban User: {selectedUserData.name || selectedUserData.email}
            </h3>

            <div className="mb-4">
              <label className="mb-2 block text-sm font-medium text-gray-700">
                Reason for ban *
              </label>
              <textarea
                value={banReason}
                onChange={(e) => setBanReason(e.target.value)}
                placeholder="Please provide a detailed reason for banning this user..."
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-red-500 focus:ring-1 focus:ring-red-500 focus:outline-none"
                rows={3}
                maxLength={500}
              />
              <div className="mt-1 text-xs text-gray-500">
                {banReason.length}/500 characters
              </div>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowBanModal(false);
                  setSelectedUser(null);
                  setBanReason("");
                }}
                className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => handleBanUser(selectedUserData.email!)}
                disabled={!banReason.trim() || isBanning}
                className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:opacity-50"
              >
                {isBanning ? "Banning..." : "Ban User"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
