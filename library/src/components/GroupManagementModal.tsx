"use client";

import { FC, useState, useEffect } from "react";
import { useAction } from "next-safe-action/hooks";
import { X, Users, UserPlus, Trash2, Globe, Lock } from "lucide-react";

import { manageGroupMembershipAction } from "@/groups/actions/manageGroupMembershipAction";

interface GroupMember {
  id: string;
  user: {
    name: string | null;
    email: string | null;
  };
  role: string | null;
  joinedAt: Date;
  isActive: boolean;
}

interface GroupManagementModalProps {
  isOpen: boolean;
  onClose: () => void;
  group: {
    id: string;
    name: string;
    description: string | null;
    isPublic: boolean;
    members?: GroupMember[];
  };
  currentUserRole?: string | null;
}

export const GroupManagementModal: FC<GroupManagementModalProps> = ({
  isOpen,
  onClose,
  group,
  currentUserRole,
}) => {
  const [activeTab, setActiveTab] = useState<"members" | "add">("members");
  const [localMembers, setLocalMembers] = useState<GroupMember[]>(
    group.members || []
  );
  const [newMemberData, setNewMemberData] = useState({
    userEmail: "",
    role: "member",
  });

  // Update local members when group prop changes
  useEffect(() => {
    setLocalMembers(group.members || []);
  }, [group.members]);

  const { execute: manageMembership, isExecuting } = useAction(
    manageGroupMembershipAction,
    {
      onSuccess: (result) => {
        if (activeTab === "add") {
          // Optimistically add the new member to local state
          const newMember: GroupMember = {
            id: `temp_${Date.now()}`, // Temporary ID until page refresh
            user: {
              name: newMemberData.userEmail.split("@")[0] || null,
              email: newMemberData.userEmail,
            },
            role: newMemberData.role || "member",
            joinedAt: new Date(),
            isActive: true,
          };

          setLocalMembers((prev) => [...prev, newMember]);

          setNewMemberData({
            userEmail: "",
            role: "member",
          });
          setActiveTab("members");
        }
      },
      onError: (error) => {
        console.error("Error managing membership:", error);
      },
    }
  );

  const handleAddMember = (e: React.FormEvent) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("groupId", group.id);
    formData.append("userEmail", newMemberData.userEmail);
    formData.append("role", newMemberData.role);
    formData.append("action", "add");

    manageMembership(formData);
  };

  const handleRemoveMember = (memberEmail: string) => {
    if (window.confirm("Are you sure you want to remove this member?")) {
      // Optimistically remove from local state
      setLocalMembers((prev) =>
        prev.filter((member) => member.user.email !== memberEmail)
      );

      const formData = new FormData();
      formData.append("groupId", group.id);
      formData.append("userEmail", memberEmail);
      formData.append("action", "remove");

      manageMembership(formData);
    }
  };

  const isAdmin = currentUserRole === "admin";

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="max-h-[80vh] w-full max-w-2xl overflow-hidden rounded-lg bg-white shadow-xl">
        {/* Header */}
        <div className="border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Users className="h-5 w-5 text-blue-600" />
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  {group.name}
                </h2>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  {group.isPublic ? (
                    <>
                      <Globe className="h-3 w-3" />
                      Public Group
                    </>
                  ) : (
                    <>
                      <Lock className="h-3 w-3" />
                      Private Group
                    </>
                  )}
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="rounded-lg p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {group.description && (
            <p className="mt-2 text-sm text-gray-600">{group.description}</p>
          )}

          {isAdmin && (
            <div className="mt-4 flex gap-2">
              <button
                onClick={() => setActiveTab("members")}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  activeTab === "members"
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                Members ({localMembers.length})
              </button>
              <button
                onClick={() => setActiveTab("add")}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  activeTab === "add"
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                Add Member
              </button>
            </div>
          )}
        </div>

        {/* Content */}
        <div
          className="overflow-y-auto p-6"
          style={{ maxHeight: "calc(80vh - 140px)" }}
        >
          {activeTab === "members" && (
            <div className="space-y-4">
              {localMembers.length > 0 ? (
                localMembers.map((member) => (
                  <div
                    key={member.id}
                    className="flex items-center justify-between rounded-lg border border-gray-200 p-4"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-sm font-medium text-blue-700">
                        {member.user.name
                          ?.split(" ")
                          .map((n) => n[0])
                          .join("")
                          .toUpperCase() ||
                          member.user.email
                            ?.split("@")[0]?.[0]
                            ?.toUpperCase() ||
                          "?"}
                      </div>
                      <div>
                        <div className="font-medium text-gray-900">
                          {member.user.name ||
                            member.user.email?.split("@")[0] ||
                            "Unknown"}
                        </div>
                        <div className="text-sm text-gray-500">
                          {member.user.email}
                        </div>
                        {member.role && (
                          <div className="mt-1 text-xs text-gray-400">
                            {member.role}
                          </div>
                        )}
                      </div>
                    </div>

                    {isAdmin && member.user.email && (
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleRemoveMember(member.user.email!)}
                          disabled={isExecuting}
                          className="rounded-md p-1.5 text-gray-400 transition-colors hover:bg-red-50 hover:text-red-600 disabled:opacity-50"
                          title="Remove member"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="py-8 text-center text-gray-500">
                  <Users className="mx-auto h-12 w-12 text-gray-300" />
                  <p className="mt-2">No members found</p>
                </div>
              )}
            </div>
          )}

          {activeTab === "add" && isAdmin && (
            <form onSubmit={handleAddMember} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  User Email *
                </label>
                <input
                  type="email"
                  value={newMemberData.userEmail}
                  onChange={(e) =>
                    setNewMemberData((prev) => ({
                      ...prev,
                      userEmail: e.target.value,
                    }))
                  }
                  required
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                  placeholder="user@example.com"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Role
                </label>
                <select
                  value={newMemberData.role}
                  onChange={(e) =>
                    setNewMemberData((prev) => ({
                      ...prev,
                      role: e.target.value,
                    }))
                  }
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                >
                  <option value="member">Member</option>
                  <option value="admin">Admin</option>
                </select>
              </div>

              <div className="flex justify-end gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setActiveTab("members")}
                  className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!newMemberData.userEmail || isExecuting}
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
                >
                  {isExecuting ? "Adding..." : "Add Member"}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};
