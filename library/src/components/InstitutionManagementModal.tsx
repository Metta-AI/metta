"use client";

import { FC, useState, useEffect } from "react";
import { useAction } from "next-safe-action/hooks";
import { X, Users, Mail, UserPlus, Edit2, Trash2 } from "lucide-react";

import { manageUserMembershipAction } from "@/institutions/actions/manageUserMembershipAction";

interface InstitutionMember {
  id: string;
  user: {
    name: string | null;
    email: string | null;
  };
  role: string | null;
  department: string | null;
  title: string | null;
  joinedAt: Date;
  isActive: boolean;
}

interface InstitutionManagementModalProps {
  isOpen: boolean;
  onClose: () => void;
  institution: {
    id: string;
    name: string;
    members?: InstitutionMember[];
  };
  currentUserRole?: string | null;
}

export const InstitutionManagementModal: FC<
  InstitutionManagementModalProps
> = ({ isOpen, onClose, institution, currentUserRole }) => {
  const [activeTab, setActiveTab] = useState<"members" | "add">("members");
  const [localMembers, setLocalMembers] = useState<InstitutionMember[]>(
    institution.members || []
  );
  const [newMemberData, setNewMemberData] = useState({
    userEmail: "",
    role: "member",
    department: "",
    title: "",
  });

  // Update local members when institution prop changes
  useEffect(() => {
    setLocalMembers(institution.members || []);
  }, [institution.members]);

  const { execute: manageMembership, isExecuting } = useAction(
    manageUserMembershipAction,
    {
      onSuccess: (result) => {
        if (activeTab === "add") {
          // Optimistically add the new member to local state using current form data
          const newMember: InstitutionMember = {
            id: `temp_${Date.now()}`, // Temporary ID until page refresh
            user: {
              name: newMemberData.userEmail.split("@")[0] || null,
              email: newMemberData.userEmail,
            },
            role: newMemberData.role || "member",
            department: newMemberData.department || null,
            title: newMemberData.title || null,
            joinedAt: new Date(),
            isActive: true,
          };

          setLocalMembers((prev) => [...prev, newMember]);

          setNewMemberData({
            userEmail: "",
            role: "member",
            department: "",
            title: "",
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
    formData.append("institutionId", institution.id);
    formData.append("userEmail", newMemberData.userEmail);
    formData.append("role", newMemberData.role);
    formData.append("department", newMemberData.department);
    formData.append("title", newMemberData.title);
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
      formData.append("institutionId", institution.id);
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
              <h2 className="text-lg font-semibold text-gray-900">
                Manage {institution.name}
              </h2>
            </div>
            <button
              onClick={onClose}
              className="rounded-lg p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

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
                        {(member.role || member.department || member.title) && (
                          <div className="mt-1 text-xs text-gray-400">
                            {[member.role, member.department, member.title]
                              .filter(Boolean)
                              .join(" • ")}
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
                  <option value="researcher">Researcher</option>
                  <option value="student">Student</option>
                  <option value="faculty">Faculty</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Department
                </label>
                <input
                  type="text"
                  value={newMemberData.department}
                  onChange={(e) =>
                    setNewMemberData((prev) => ({
                      ...prev,
                      department: e.target.value,
                    }))
                  }
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                  placeholder="e.g., Computer Science"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Title
                </label>
                <input
                  type="text"
                  value={newMemberData.title}
                  onChange={(e) =>
                    setNewMemberData((prev) => ({
                      ...prev,
                      title: e.target.value,
                    }))
                  }
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                  placeholder="e.g., Senior Researcher"
                />
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
