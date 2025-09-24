"use client";

import React, { FC, useState, useEffect } from "react";
import { Users, Shield, Search, Plus, Trash2, UserCheck } from "lucide-react";
import { useAction } from "next-safe-action/hooks";

import { manageInstitutionOwnershipAction } from "@/institutions/actions/manageInstitutionOwnershipAction";
import { toggleApprovalRequirementAction } from "@/institutions/actions/toggleApprovalRequirementAction";
import { approveRejectMembershipAction } from "@/institutions/actions/approveRejectMembershipAction";

type AdminUser = {
  id: string;
  role: string;
  joinedAt: Date;
  user: {
    id: string;
    name: string | null;
    email: string | null;
  };
};

type AdminInstitution = {
  id: string;
  name: string;
  domain: string | null;
  type: string;
  isVerified: boolean;
  requiresApproval: boolean;
  createdAt: Date;
  memberCount: number;
  pendingCount: number;
  admins: AdminUser[];
  pendingRequests: AdminUser[];
};

export const AdminInstitutionsView: FC = () => {
  const [institutions, setInstitutions] = useState<AdminInstitution[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedInstitution, setSelectedInstitution] = useState<string | null>(
    null
  );
  const [actionData, setActionData] = useState({
    userEmail: "",
    action: "assign_admin" as const,
  });

  // Load institutions
  useEffect(() => {
    loadInstitutions();
  }, []);

  const loadInstitutions = async () => {
    try {
      const response = await fetch("/api/admin/institutions");
      if (response.ok) {
        const data = await response.json();
        setInstitutions(data.institutions);
      }
    } catch (error) {
      console.error("Failed to load institutions:", error);
    } finally {
      setLoading(false);
    }
  };

  // Manage ownership action
  const { execute: manageOwnership, isExecuting } = useAction(
    manageInstitutionOwnershipAction,
    {
      onSuccess: () => {
        setActionData({ userEmail: "", action: "assign_admin" });
        setSelectedInstitution(null);
        loadInstitutions(); // Reload data
      },
      onError: (error) => {
        console.error("Error managing ownership:", error);
        alert(error.error?.serverError || "Failed to update ownership");
      },
    }
  );

  // Toggle approval requirement action
  const { execute: toggleApproval, isExecuting: isTogglingApproval } =
    useAction(toggleApprovalRequirementAction, {
      onSuccess: () => {
        loadInstitutions(); // Reload data
      },
      onError: (error) => {
        console.error("Error toggling approval:", error);
        alert(error.error?.serverError || "Failed to update approval setting");
      },
    });

  // Approve/reject membership action
  const {
    execute: approveRejectMembership,
    isExecuting: isApprovingRejecting,
  } = useAction(approveRejectMembershipAction, {
    onSuccess: () => {
      loadInstitutions(); // Reload data
    },
    onError: (error) => {
      console.error("Error processing membership request:", error);
      alert(error.error?.serverError || "Failed to process membership request");
    },
  });

  const handleOwnershipAction = (
    institutionId: string,
    userEmail: string,
    action: string
  ) => {
    const formData = new FormData();
    formData.append("institutionId", institutionId);
    formData.append("userEmail", userEmail);
    formData.append("action", action);
    manageOwnership(formData);
  };

  const handleApprovalToggle = (
    institutionId: string,
    requiresApproval: boolean
  ) => {
    const formData = new FormData();
    formData.append("institutionId", institutionId);
    formData.append("requiresApproval", requiresApproval.toString());
    toggleApproval(formData);
  };

  const handleMembershipAction = (
    institutionId: string,
    userEmail: string,
    action: "approve" | "reject"
  ) => {
    const formData = new FormData();
    formData.append("institutionId", institutionId);
    formData.append("userEmail", userEmail);
    formData.append("action", action);
    approveRejectMembership(formData);
  };

  const filteredInstitutions = institutions.filter(
    (institution) =>
      institution.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      institution.domain?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatDate = (date: Date | string) => {
    return new Date(date).toLocaleDateString();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-gray-600">Loading institutions...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="relative">
        <Search className="absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2 text-gray-400" />
        <input
          type="text"
          placeholder="Search institutions..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full rounded-lg border border-gray-300 py-3 pr-4 pl-10 text-gray-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
        />
      </div>

      {/* Quick Add Owner */}
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        <h3 className="mb-4 text-lg font-semibold text-gray-900">
          Quick Actions
        </h3>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
          <select
            value={selectedInstitution || ""}
            onChange={(e) => setSelectedInstitution(e.target.value || null)}
            className="rounded-md border border-gray-300 px-3 py-2 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
          >
            <option value="">Select Institution</option>
            {institutions.map((institution) => (
              <option key={institution.id} value={institution.id}>
                {institution.name}
              </option>
            ))}
          </select>

          <input
            type="email"
            placeholder="User email"
            value={actionData.userEmail}
            onChange={(e) =>
              setActionData((prev) => ({ ...prev, userEmail: e.target.value }))
            }
            className="rounded-md border border-gray-300 px-3 py-2 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
          />

          <select
            value={actionData.action}
            onChange={(e) =>
              setActionData((prev) => ({
                ...prev,
                action: e.target.value as any,
              }))
            }
            className="rounded-md border border-gray-300 px-3 py-2 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
          >
            <option value="assign_admin">Assign Admin</option>
            <option value="remove_admin">Remove Admin</option>
          </select>

          <button
            onClick={() =>
              selectedInstitution &&
              actionData.userEmail &&
              handleOwnershipAction(
                selectedInstitution,
                actionData.userEmail,
                actionData.action
              )
            }
            disabled={
              isExecuting || !selectedInstitution || !actionData.userEmail
            }
            className="flex items-center justify-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:bg-gray-400"
          >
            <UserCheck className="h-4 w-4" />
            {isExecuting ? "Processing..." : "Execute"}
          </button>
        </div>
      </div>

      {/* Institutions List */}
      <div className="space-y-4">
        {filteredInstitutions.map((institution) => (
          <div
            key={institution.id}
            className="rounded-lg border border-gray-200 bg-white p-6"
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xl font-semibold text-gray-900">
                    {institution.name}
                  </h3>
                  {institution.isVerified && (
                    <UserCheck className="h-5 w-5 text-green-600" />
                  )}
                </div>
                <div className="mt-1 space-y-1 text-sm text-gray-600">
                  <p>Domain: {institution.domain || "None"}</p>
                  <p>Type: {institution.type}</p>
                  <p>Members: {institution.memberCount}</p>
                  <p>Created: {formatDate(institution.createdAt)}</p>
                </div>
              </div>
            </div>

            {/* Approval Settings */}
            <div className="mt-4 rounded-lg bg-gray-50 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-gray-900">
                    Membership Approval
                  </h4>
                  <p className="text-sm text-gray-600">
                    {institution.requiresApproval
                      ? "New members require admin approval"
                      : "Users can join automatically"}
                  </p>
                </div>
                <label className="relative inline-flex cursor-pointer items-center">
                  <input
                    type="checkbox"
                    className="peer sr-only"
                    checked={institution.requiresApproval}
                    onChange={(e) =>
                      handleApprovalToggle(institution.id, e.target.checked)
                    }
                    disabled={isTogglingApproval}
                  />
                  <div className="peer h-6 w-11 rounded-full bg-gray-200 peer-checked:bg-blue-600 peer-focus:ring-4 peer-focus:ring-blue-300 peer-focus:outline-none peer-disabled:opacity-50 after:absolute after:top-[2px] after:left-[2px] after:h-5 after:w-5 after:rounded-full after:border after:border-gray-300 after:bg-white after:transition-all after:content-[''] peer-checked:after:translate-x-full peer-checked:after:border-white"></div>
                </label>
              </div>
            </div>

            {/* Pending Requests */}
            {institution.pendingRequests.length > 0 && (
              <div className="mt-4">
                <h4 className="mb-3 flex items-center gap-2 font-medium text-gray-900">
                  <UserCheck className="h-4 w-4 text-orange-600" />
                  Pending Requests ({institution.pendingRequests.length})
                </h4>
                <div className="space-y-2">
                  {institution.pendingRequests.map((request) => (
                    <div
                      key={request.id}
                      className="flex items-center justify-between rounded-md bg-orange-50 p-3"
                    >
                      <div>
                        <div className="font-medium text-gray-900">
                          {request.user.name || request.user.email}
                        </div>
                        <div className="text-sm text-gray-600">
                          {request.user.email} • Requested{" "}
                          {formatDate(request.joinedAt)}
                          {request.department && ` • ${request.department}`}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() =>
                            handleMembershipAction(
                              institution.id,
                              request.user.email!,
                              "approve"
                            )
                          }
                          disabled={isApprovingRejecting}
                          className="rounded bg-green-600 px-3 py-1 text-sm text-white hover:bg-green-700 disabled:opacity-50"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() =>
                            handleMembershipAction(
                              institution.id,
                              request.user.email!,
                              "reject"
                            )
                          }
                          disabled={isApprovingRejecting}
                          className="rounded bg-red-600 px-3 py-1 text-sm text-white hover:bg-red-700 disabled:opacity-50"
                        >
                          Reject
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Admins */}
            <div className="mt-6">
              {/* Admins */}
              <div>
                <h4 className="mb-3 flex items-center gap-2 font-medium text-gray-900">
                  <Shield className="h-4 w-4 text-blue-600" />
                  Admins ({institution.admins.length})
                </h4>
                <div className="space-y-2">
                  {institution.admins.length === 0 ? (
                    <p className="text-sm text-gray-500">No admins assigned</p>
                  ) : (
                    institution.admins.map((admin) => (
                      <div
                        key={admin.id}
                        className="flex items-center justify-between rounded-md bg-blue-50 p-3"
                      >
                        <div>
                          <div className="font-medium text-gray-900">
                            {admin.user.name || admin.user.email}
                          </div>
                          <div className="text-sm text-gray-600">
                            {admin.user.email} • Since{" "}
                            {formatDate(admin.joinedAt)}
                          </div>
                        </div>
                        <button
                          onClick={() =>
                            handleOwnershipAction(
                              institution.id,
                              admin.user.email!,
                              "remove_admin"
                            )
                          }
                          disabled={isExecuting}
                          className="text-red-600 hover:text-red-800"
                          title="Remove admin"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredInstitutions.length === 0 && (
        <div className="py-12 text-center">
          <Users className="mx-auto mb-4 h-12 w-12 text-gray-400" />
          <p className="text-gray-600">
            {searchQuery
              ? "No institutions found matching your search."
              : "No institutions found."}
          </p>
        </div>
      )}
    </div>
  );
};
