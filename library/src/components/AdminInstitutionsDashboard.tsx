"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";
import { useAction } from "next-safe-action/hooks";

import { AdminDataTable, type ColumnDef } from "@/components/AdminDataTable";
import { SectionHeading } from "@/components/ui/section-heading";
import { Stat } from "@/components/ui/stat";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { getInitialsFromName } from "@/lib/utils/text";
import { formatDate } from "@/lib/utils/date";
import { useErrorHandling } from "@/lib/hooks/useErrorHandling";
import { manageInstitutionOwnershipAction } from "@/institutions/actions/manageInstitutionOwnershipAction";
import { toggleApprovalRequirementAction } from "@/institutions/actions/toggleApprovalRequirementAction";
import { approveRejectMembershipAction } from "@/institutions/actions/approveRejectMembershipAction";
import { listAdminInstitutions } from "@/lib/api/resources/admin";

interface AdminInstitutionUser {
  id: string;
  role: string;
  joinedAt: Date;
  department: string | null;
  title: string | null;
  user: {
    id: string;
    name: string | null;
    email: string | null;
  };
}

export interface AdminInstitutionRow {
  id: string;
  name: string;
  domain: string | null;
  type: string;
  isVerified: boolean;
  requiresApproval: boolean;
  createdAt: Date;
  memberCount: number;
  pendingCount: number;
  admins: AdminInstitutionUser[];
  pendingRequests: AdminInstitutionUser[];
}

interface AdminInstitutionsDashboardProps {
  initialInstitutions: AdminInstitutionRow[];
}

export const AdminInstitutionsDashboard: React.FC<
  AdminInstitutionsDashboardProps
> = ({ initialInstitutions }) => {
  const [institutions, setInstitutions] = useState(initialInstitutions);
  const [searchQuery, setSearchQuery] = useState(" ");
  const [selectedInstitutionId, setSelectedInstitutionId] = useState<
    string | null
  >(initialInstitutions[0]?.id ?? null);
  const [actionData, setActionData] = useState({
    userEmail: "",
    action: "assign_admin" as const,
  });

  const {
    error: ownershipError,
    setError: setOwnershipError,
    clearError: clearOwnershipError,
  } = useErrorHandling({ fallbackMessage: "Failed to update ownership" });

  const { execute: manageOwnership, isExecuting: isManagingOwnership } =
    useAction(manageInstitutionOwnershipAction, {
      onSuccess: () => {
        setActionData({ userEmail: "", action: "assign_admin" });
        setSelectedInstitutionId(null);
        void refreshInstitutions();
        clearOwnershipError();
      },
      onError: (error) => setOwnershipError(error),
    });

  const { execute: toggleApproval, isExecuting: isTogglingApproval } =
    useAction(toggleApprovalRequirementAction, {
      onSuccess: () => void refreshInstitutions(),
    });

  const { execute: approveRejectMembership, isExecuting: isProcessingRequest } =
    useAction(approveRejectMembershipAction, {
      onSuccess: () => void refreshInstitutions(),
    });

  const refreshInstitutions = useCallback(async () => {
    try {
      const data = await listAdminInstitutions();
      // Map AdminInstitution to AdminInstitutionRow
      const mapped: AdminInstitutionRow[] = data.map((inst) => ({
        ...inst,
        type: "University" as const, // Default type
        isVerified: true,
        createdAt: new Date(),
        pendingCount: inst.pendingMembersCount,
        admins: [], // Will be loaded separately
        pendingRequests: [], // Will be loaded separately
      }));
      setInstitutions(mapped);
    } catch (error) {
      console.error("Failed to refresh institutions:", error);
    }
  }, []);

  useEffect(() => {
    void refreshInstitutions();
  }, [refreshInstitutions]);

  useEffect(() => {
    if (!selectedInstitutionId && institutions.length > 0) {
      setSelectedInstitutionId(institutions[0].id);
    }
  }, [institutions, selectedInstitutionId]);

  const filteredInstitutions = useMemo(() => {
    const normalized = searchQuery.trim().toLowerCase();
    const items = normalized
      ? institutions.filter((institution) =>
          [institution.name, institution.domain ?? "", institution.type]
            .join(" ")
            .toLowerCase()
            .includes(normalized)
        )
      : institutions;

    return [...items].sort((a, b) => a.name.localeCompare(b.name));
  }, [institutions, searchQuery]);

  const selectedInstitution = useMemo(
    () =>
      selectedInstitutionId
        ? (institutions.find(
            (institution) => institution.id === selectedInstitutionId
          ) ?? null)
        : null,
    [institutions, selectedInstitutionId]
  );

  const columns = useMemo<ColumnDef<AdminInstitutionRow, unknown>[]>(
    () => [
      {
        accessorKey: "name",
        header: "Institution",
        cell: ({ row }) => {
          const institution = row.original;
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
        sortingFn: "text",
      },
      {
        accessorKey: "memberCount",
        header: "Members",
        cell: ({ row }) => row.original.memberCount,
      },
      {
        accessorKey: "pendingCount",
        header: "Pending",
        cell: ({ row }) => row.original.pendingRequests.length,
      },
      {
        accessorKey: "admins",
        header: "Admins",
        cell: ({ row }) => row.original.admins.length,
      },
      {
        accessorKey: "requiresApproval",
        header: "Approval",
        cell: ({ row }) => (
          <span
            className={cn(
              "inline-flex items-center rounded-full px-2 py-1 text-xs font-medium",
              row.original.requiresApproval
                ? "bg-amber-100 text-amber-800"
                : "bg-emerald-100 text-emerald-700"
            )}
          >
            {row.original.requiresApproval ? "Required" : "Open"}
          </span>
        ),
      },
    ],
    [isTogglingApproval, toggleApproval]
  );

  const handleOwnershipAction = useCallback(
    (
      institutionId: string,
      userEmail: string,
      action: "assign_admin" | "remove_admin"
    ) => {
      const formData = new FormData();
      formData.append("institutionId", institutionId);
      formData.append("userEmail", userEmail);
      formData.append("action", action);
      manageOwnership(formData);
    },
    [manageOwnership]
  );

  const handleMembershipAction = useCallback(
    (
      institutionId: string,
      userEmail: string,
      action: "approve" | "reject"
    ) => {
      const formData = new FormData();
      formData.append("institutionId", institutionId);
      formData.append("userEmail", userEmail);
      formData.append("action", action);
      approveRejectMembership(formData);
    },
    [approveRejectMembership]
  );

  return (
    <div className="space-y-6">
      <div className="relative">
        <Search className="text-muted-foreground absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2" />
        <input
          type="text"
          placeholder="Search institutions..."
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          className="border-border text-foreground focus:border-primary focus:ring-primary/20 w-full rounded-lg border py-3 pr-4 pl-10 text-sm focus:ring-2 focus:outline-none"
        />
        {ownershipError && (
          <p className="text-destructive mt-2 text-sm">{ownershipError}</p>
        )}
      </div>

      <div className="border-border bg-card rounded-xl border">
        <div className="border-border border-b p-4">
          <SectionHeading title="Quick actions" className="gap-2" />
        </div>
        <div className="grid grid-cols-1 gap-4 p-4 md:grid-cols-4">
          <select
            value={selectedInstitutionId ?? ""}
            onChange={(event) =>
              setSelectedInstitutionId(event.target.value || null)
            }
            className="border-border focus:border-primary focus:ring-primary/50 rounded-md border px-3 py-2 text-sm focus:ring-1 focus:outline-none"
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
            onChange={(event) =>
              setActionData((prev) => ({
                ...prev,
                userEmail: event.target.value,
              }))
            }
            className="border-border focus:border-primary focus:ring-primary/50 rounded-md border px-3 py-2 text-sm focus:ring-1 focus:outline-none"
          />

          <select
            value={actionData.action}
            onChange={(event) =>
              setActionData((prev) => ({
                ...prev,
                action: event.target.value as typeof prev.action,
              }))
            }
            className="border-border focus:border-primary focus:ring-primary/50 rounded-md border px-3 py-2 text-sm focus:ring-1 focus:outline-none"
          >
            <option value="assign_admin">Assign Admin</option>
            <option value="remove_admin">Remove Admin</option>
          </select>

          <Button
            onClick={() =>
              selectedInstitutionId &&
              actionData.userEmail &&
              handleOwnershipAction(
                selectedInstitutionId,
                actionData.userEmail,
                actionData.action
              )
            }
            disabled={
              isManagingOwnership ||
              !selectedInstitutionId ||
              !actionData.userEmail
            }
            className="flex items-center justify-center gap-2"
          >
            {isManagingOwnership ? "Processing..." : "Execute"}
          </Button>
        </div>
      </div>

      <AdminDataTable
        data={filteredInstitutions}
        columns={columns}
        isLoading={false}
        emptyMessage={
          searchQuery
            ? "No institutions match your search."
            : "No institutions found."
        }
        onRowClick={(institution) => setSelectedInstitutionId(institution.id)}
        initialSorting={[{ id: "name", desc: false }]}
      />

      {selectedInstitution ? (
        <div className="border-border bg-card rounded-xl border">
          <div className="border-border border-b p-4">
            <SectionHeading
              title={selectedInstitution.name}
              description={`Domain: ${selectedInstitution.domain ?? "None"}`}
              actions={
                selectedInstitution.requiresApproval ? (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      const formData = new FormData();
                      formData.append("institutionId", selectedInstitution.id);
                      formData.append("requiresApproval", "false");
                      toggleApproval(formData);
                    }}
                    disabled={isTogglingApproval}
                  >
                    Disable approval
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    onClick={() => {
                      const formData = new FormData();
                      formData.append("institutionId", selectedInstitution.id);
                      formData.append("requiresApproval", "true");
                      toggleApproval(formData);
                    }}
                    disabled={isTogglingApproval}
                  >
                    Require approval
                  </Button>
                )
              }
            />
          </div>

          <div className="space-y-6 p-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="Members" value={selectedInstitution.memberCount} />
              <Stat
                label="Pending"
                value={selectedInstitution.pendingRequests.length}
                helperText={
                  selectedInstitution.pendingRequests.length > 0
                    ? "Review approvals"
                    : "All caught up"
                }
              />
              <Stat label="Admins" value={selectedInstitution.admins.length} />
              <Stat
                label="Created"
                value={formatDate(selectedInstitution.createdAt)}
              />
            </div>

            {selectedInstitution.pendingRequests.length > 0 && (
              <div className="space-y-3">
                <SectionHeading
                  title={`Pending Requests (${selectedInstitution.pendingRequests.length})`}
                />
                <div className="space-y-2">
                  {selectedInstitution.pendingRequests.map((request) => (
                    <div
                      key={request.id}
                      className="border-border bg-muted/40 flex flex-col gap-3 rounded-lg border p-3 md:flex-row md:items-center md:justify-between"
                    >
                      <div>
                        <div className="text-foreground font-medium">
                          {request.user.name || request.user.email}
                        </div>
                        <div className="text-muted-foreground text-sm">
                          {request.user.email} • Requested{" "}
                          {formatDate(request.joinedAt)}
                          {request.department && ` • ${request.department}`}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            handleMembershipAction(
                              selectedInstitution.id,
                              request.user.email!,
                              "approve"
                            )
                          }
                          disabled={isProcessingRequest}
                        >
                          Approve
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() =>
                            handleMembershipAction(
                              selectedInstitution.id,
                              request.user.email!,
                              "reject"
                            )
                          }
                          disabled={isProcessingRequest}
                        >
                          Reject
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="space-y-3">
              <SectionHeading
                title={`Admins (${selectedInstitution.admins.length})`}
                description="Manage institution administrators"
              />
              <div className="space-y-2">
                {selectedInstitution.admins.length === 0 ? (
                  <p className="text-muted-foreground text-sm">
                    No admins assigned
                  </p>
                ) : (
                  selectedInstitution.admins.map((admin) => (
                    <div
                      key={admin.id}
                      className="border-border bg-muted/30 flex items-center justify-between rounded-lg border p-3"
                    >
                      <div>
                        <div className="text-foreground font-medium">
                          {admin.user.name || admin.user.email}
                        </div>
                        <div className="text-muted-foreground text-sm">
                          {admin.user.email} • Since{" "}
                          {formatDate(admin.joinedAt)}
                        </div>
                      </div>
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() =>
                          handleOwnershipAction(
                            selectedInstitution.id,
                            admin.user.email!,
                            "remove_admin"
                          )
                        }
                        disabled={isManagingOwnership}
                      >
                        Remove
                      </Button>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="border-border bg-card text-muted-foreground space-y-4 rounded-xl border p-6 text-center">
          Select an institution to view details.
        </div>
      )}
    </div>
  );
};
