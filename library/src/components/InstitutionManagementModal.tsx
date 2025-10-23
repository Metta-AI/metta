"use client";

import React, { FC, useEffect, useMemo, useState, useRef } from "react";
import { useAction } from "next-safe-action/hooks";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { X, Users, Mail, UserPlus, Edit2, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { manageUserMembershipAction } from "@/institutions/actions/manageUserMembershipAction";
import { toggleApprovalRequirementAction } from "@/institutions/actions/toggleApprovalRequirementAction";
import { useErrorHandling } from "@/lib/hooks/useErrorHandling";
import { getUserDisplayName } from "@/lib/utils/user";
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
import { Textarea } from "@/components/ui/textarea";

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
    requiresApproval?: boolean;
    members?: InstitutionMember[];
  };
  currentUserRole?: string | null;
}

const addMemberSchema = z.object({
  userEmail: z.string().email("Valid email is required"),
  role: z.string().min(1),
  department: z
    .string()
    .optional()
    .transform((value) => value?.trim() || undefined),
  title: z
    .string()
    .optional()
    .transform((value) => value?.trim() || undefined),
});

type AddMemberValues = z.infer<typeof addMemberSchema>;

const addMemberDefaults: AddMemberValues = {
  userEmail: "",
  role: "member",
  department: undefined,
  title: undefined,
};

export const InstitutionManagementModal: FC<
  InstitutionManagementModalProps
> = ({ isOpen, onClose, institution, currentUserRole }) => {
  const [activeTab, setActiveTab] = useState<"members" | "add" | "settings">(
    "members"
  );
  const [localMembers, setLocalMembers] = useState<InstitutionMember[]>(
    institution.members || []
  );
  const pendingMembershipActionRef = useRef<"add" | "remove" | null>(null);
  const pendingMemberEmailRef = useRef<string | null>(null);
  const pendingApprovalRef = useRef<boolean | null>(null);

  const form = useForm<AddMemberValues>({
    resolver: zodResolver(addMemberSchema),
    defaultValues: addMemberDefaults,
  });

  const {
    error: membershipError,
    setError: setMembershipError,
    clearError: clearMembershipError,
  } = useErrorHandling({
    fallbackMessage: "Failed to manage membership. Please try again.",
  });

  useEffect(() => {
    setLocalMembers(institution.members || []);
  }, [institution.members]);

  useEffect(() => {
    if (!isOpen) {
      setActiveTab("members");
      form.reset(addMemberDefaults);
      clearMembershipError();
    }
  }, [isOpen, form, clearMembershipError]);

  const { execute: manageMembership, isExecuting } = useAction(
    manageUserMembershipAction,
    {
      onSuccess: () => {
        const pendingAction = pendingMembershipActionRef.current;
        if (pendingAction === "add") {
          const values = form.getValues();
          const newMember: InstitutionMember = {
            id: `temp_${Date.now()}`,
            user: {
              name: values.userEmail.split("@")[0] || null,
              email: values.userEmail,
            },
            role: values.role || "member",
            department: values.department || null,
            title: values.title || null,
            joinedAt: new Date(),
            isActive: true,
          };

          setLocalMembers((prev) => [...prev, newMember]);
          form.reset(addMemberDefaults);
          clearMembershipError();
          setActiveTab("members");
          toast.success("Invitation sent");
        }

        if (pendingAction === "remove" && pendingMemberEmailRef.current) {
          setLocalMembers((prev) =>
            prev.filter(
              (member) => member.user.email !== pendingMemberEmailRef.current
            )
          );
          toast.success("Member removed");
        }

        pendingMembershipActionRef.current = null;
        pendingMemberEmailRef.current = null;
      },
      onError: (error) => {
        console.error("Error managing membership:", error);
        setMembershipError(error);
        const errorMessage =
          error.error?.serverError ?? "Failed to manage membership";
        toast.error(errorMessage);
        pendingMembershipActionRef.current = null;
        pendingMemberEmailRef.current = null;
      },
    }
  );

  const { execute: toggleApproval, isExecuting: isTogglingApproval } =
    useAction(toggleApprovalRequirementAction, {
      onSuccess: () => {
        const requiresApproval = pendingApprovalRef.current;
        if (requiresApproval !== null) {
          toast.success(
            requiresApproval
              ? "Membership approval now required"
              : "Membership approval disabled"
          );
        }
        pendingApprovalRef.current = null;
      },
      onError: (error) => {
        console.error("Error toggling approval:", error);
        toast.error("Failed to update approval setting");
        pendingApprovalRef.current = null;
      },
    });

  const handleApprovalToggle = (requiresApproval: boolean) => {
    pendingApprovalRef.current = requiresApproval;
    const formData = new FormData();
    formData.append("institutionId", institution.id);
    formData.append("requiresApproval", requiresApproval.toString());
    toggleApproval(formData);
  };

  const handleRemoveMember = (memberEmail: string) => {
    pendingMembershipActionRef.current = "remove";
    pendingMemberEmailRef.current = memberEmail;
    const formData = new FormData();
    formData.append("institutionId", institution.id);
    formData.append("userEmail", memberEmail);
    formData.append("action", "remove");

    manageMembership(formData);
  };

  const handleAddMember = (values: AddMemberValues) => {
    pendingMembershipActionRef.current = "add";
    const formData = new FormData();
    formData.append("institutionId", institution.id);
    formData.append("userEmail", values.userEmail);
    formData.append("role", values.role);
    formData.append("department", values.department ?? "");
    formData.append("title", values.title ?? "");
    formData.append("action", "add");

    manageMembership(formData);
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
                Members ({memberCount})
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
              <button
                onClick={() => setActiveTab("settings")}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
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
                          <SelectItem value="researcher">Researcher</SelectItem>
                          <SelectItem value="student">Student</SelectItem>
                          <SelectItem value="faculty">Faculty</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="department"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Department</FormLabel>
                      <FormControl>
                        <Input
                          {...field}
                          value={field.value ?? ""}
                          onChange={(event) =>
                            field.onChange(event.target.value)
                          }
                          placeholder="e.g., Computer Science"
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="title"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Title</FormLabel>
                      <FormControl>
                        <Input
                          {...field}
                          value={field.value ?? ""}
                          onChange={(event) =>
                            field.onChange(event.target.value)
                          }
                          placeholder="e.g., Senior Researcher"
                        />
                      </FormControl>
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
              <div>
                <h3 className="mb-4 text-lg font-medium text-gray-900">
                  Institution Settings
                </h3>

                <div className="rounded-lg border border-gray-200 p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium text-gray-900">
                        Membership Approval
                      </h4>
                      <p className="mt-1 text-sm text-gray-600">
                        {institution.requiresApproval
                          ? "New members require admin approval before they can join"
                          : "Users can join this institution automatically"}
                      </p>
                    </div>
                    <label className="relative inline-flex cursor-pointer items-center">
                      <input
                        type="checkbox"
                        className="peer sr-only"
                        checked={institution.requiresApproval || false}
                        onChange={(e) => handleApprovalToggle(e.target.checked)}
                        disabled={isTogglingApproval}
                      />
                      <div className="peer h-6 w-11 rounded-full bg-gray-200 peer-checked:bg-blue-600 peer-focus:ring-4 peer-focus:ring-blue-300 peer-focus:outline-none peer-disabled:opacity-50 after:absolute after:top-[2px] after:left-[2px] after:h-5 after:w-5 after:rounded-full after:border after:border-gray-300 after:bg-white after:transition-all after:content-[''] peer-checked:after:translate-x-full peer-checked:after:border-white"></div>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
