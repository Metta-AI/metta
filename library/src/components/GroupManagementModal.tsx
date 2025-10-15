"use client";

import { FC, useEffect, useMemo, useState } from "react";
import { useAction } from "next-safe-action/hooks";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { X, Users, UserPlus, Trash2, Globe, Lock } from "lucide-react";

import { manageGroupMembershipAction } from "@/groups/actions/manageGroupMembershipAction";
import { useErrorHandling } from "@/lib/hooks/useErrorHandling";
import { getUserDisplayName } from "@/lib/utils/user";
import { useDeleteGroup } from "@/hooks/mutations/admin/useDeleteGroup";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

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
    institution: {
      id: string;
      name: string;
    };
    members?: GroupMember[];
  };
  currentUserRole?: string | null;
}

const addGroupMemberSchema = z.object({
  userEmail: z.string().email("Valid email is required"),
  role: z.string().min(1),
});

type AddGroupMemberValues = z.infer<typeof addGroupMemberSchema>;

const defaultAddGroupValues: AddGroupMemberValues = {
  userEmail: "",
  role: "member",
};

export const GroupManagementModal: FC<GroupManagementModalProps> = ({
  isOpen,
  onClose,
  group,
  currentUserRole,
}) => {
  const [activeTab, setActiveTab] = useState<"members" | "add" | "settings">(
    "members"
  );
  const [localMembers, setLocalMembers] = useState<GroupMember[]>(
    group.members || []
  );

  const form = useForm<AddGroupMemberValues>({
    resolver: zodResolver(addGroupMemberSchema),
    defaultValues: defaultAddGroupValues,
  });

  const {
    error: membershipError,
    setError: setMembershipError,
    clearError: clearMembershipError,
  } = useErrorHandling({
    fallbackMessage: "Failed to manage membership. Please try again.",
  });

  const deleteGroupMutation = useDeleteGroup();

  useEffect(() => {
    setLocalMembers(group.members || []);
  }, [group.members]);

  useEffect(() => {
    if (!isOpen) {
      setActiveTab("members");
      form.reset(defaultAddGroupValues);
      clearMembershipError();
    }
  }, [isOpen, form, clearMembershipError]);

  const { execute: manageMembership, isExecuting } = useAction(
    manageGroupMembershipAction,
    {
      onSuccess: () => {
        if (activeTab === "add") {
          const values = form.getValues();
          const newMember: GroupMember = {
            id: `temp_${Date.now()}`,
            user: {
              name: values.userEmail.split("@")[0] || null,
              email: values.userEmail,
            },
            role: values.role,
            joinedAt: new Date(),
            isActive: true,
          };

          setLocalMembers((prev) => [...prev, newMember]);
          form.reset(defaultAddGroupValues);
          clearMembershipError();
          setActiveTab("members");
        }
      },
      onError: (error) => {
        console.error("Error managing membership:", error);
        setMembershipError(error);
      },
    }
  );

  const handleAddMember = (values: AddGroupMemberValues) => {
    const formData = new FormData();
    formData.append("groupId", group.id);
    formData.append("userEmail", values.userEmail);
    formData.append("role", values.role);
    formData.append("action", "add");
    manageMembership(formData);
  };

  const handleRemoveMember = (memberEmail: string) => {
    if (!window.confirm("Are you sure you want to remove this member?")) {
      return;
    }

    setLocalMembers((prev) =>
      prev.filter((member) => member.user.email !== memberEmail)
    );

    const formData = new FormData();
    formData.append("groupId", group.id);
    formData.append("userEmail", memberEmail);
    formData.append("action", "remove");

    manageMembership(formData);
  };

  const handleDeleteGroup = () => {
    if (
      !window.confirm(
        `Are you sure you want to delete "${group.name}"? This action cannot be undone and will remove all members.`
      )
    ) {
      return;
    }

    const formData = new FormData();
    formData.append("groupId", group.id);

    deleteGroupMutation.mutate(
      { groupId: group.id },
      {
        onSuccess: () => {
          onClose();
        },
      }
    );
  };

  const isAdmin = currentUserRole === "admin";

  if (!isOpen) return null;

  const memberCount = useMemo(() => localMembers.length, [localMembers.length]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="max-h-[80vh] w-full max-w-2xl overflow-hidden rounded-lg bg-white shadow-xl">
        <div className="border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Users className="h-5 w-5 text-blue-600" />
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  {group.name}
                </h2>
                <div className="flex items-center gap-4 text-sm text-gray-500">
                  <div className="flex items-center gap-1">
                    {group.isPublic ? (
                      <>
                        <Globe className="h-3 w-3" /> Public Group
                      </>
                    ) : (
                      <>
                        <Lock className="h-3 w-3" /> Private Group
                      </>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <Users className="h-3 w-3" /> {group.institution.name}
                  </div>
                </div>
              </div>
            </div>
            <button
              onClick={() => {
                form.reset(defaultAddGroupValues);
                onClose();
              }}
              className="cursor-pointer rounded-lg p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {isAdmin && (
            <div className="mt-4 flex gap-2">
              <button
                onClick={() => setActiveTab("members")}
                className={`cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  activeTab === "members"
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                Members ({memberCount})
              </button>
              <button
                onClick={() => setActiveTab("add")}
                className={`cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  activeTab === "add"
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                Add Member
              </button>
              <button
                onClick={() => setActiveTab("settings")}
                className={`cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  activeTab === "settings"
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                Settings
              </button>
            </div>
          )}
        </div>

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
                          {getUserDisplayName(
                            member.user.name,
                            member.user.email
                          )}
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
                          className="cursor-pointer rounded-md p-1.5 text-gray-400 transition-colors hover:bg-red-50 hover:text-red-600 disabled:opacity-50"
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
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(handleAddMember)}
                className="space-y-4"
              >
                <FormField
                  control={form.control}
                  name="userEmail"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>User Email *</FormLabel>
                      <FormControl>
                        <Input
                          {...field}
                          type="email"
                          placeholder="user@example.com"
                          autoComplete="email"
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="role"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Role</FormLabel>
                      <Select
                        value={field.value}
                        onValueChange={field.onChange}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a role" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="member">Member</SelectItem>
                          <SelectItem value="admin">Admin</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                {membershipError && (
                  <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">
                    {membershipError}
                  </div>
                )}

                <div className="flex justify-end gap-3 pt-4">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setActiveTab("members")}
                  >
                    Cancel
                  </Button>
                  <Button type="submit" disabled={isExecuting}>
                    {isExecuting ? "Adding..." : "Add Member"}
                  </Button>
                </div>
              </form>
            </Form>
          )}

          {activeTab === "settings" && isAdmin && (
            <div className="space-y-6">
              {/* Danger Zone - Delete Group */}
              <div className="rounded-lg border border-red-200 bg-red-50 p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-red-900">
                      Delete Group
                    </h3>
                    <p className="mt-1 text-sm text-red-700">
                      Permanently delete this group and remove all members. This
                      action cannot be undone.
                    </p>
                  </div>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={handleDeleteGroup}
                    disabled={deleteGroupMutation.isPending}
                    className="ml-4"
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    {deleteGroupMutation.isPending ? "Deleting..." : "Delete"}
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
