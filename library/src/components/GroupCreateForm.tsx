"use client";

import { FC, useEffect } from "react";
import { z } from "zod";
import { X, Users, Globe, Lock, Building } from "lucide-react";

import { useFormWithMutation } from "@/lib/hooks/useFormWithMutation";
import { useCreateGroup } from "@/hooks/mutations/admin";
import { ErrorAlert } from "@/components/ui/error-alert";
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
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";

type Institution = {
  id: string;
  name: string;
};

interface GroupCreateFormProps {
  isOpen: boolean;
  onClose: () => void;
  userInstitutions: Institution[];
}

const groupSchema = z.object({
  institutionId: z.string().min(1, "Institution is required"),
  name: z
    .string()
    .min(1, "Group name is required")
    .max(100, "Group name must be 100 characters or less")
    .regex(
      /^[a-zA-Z0-9_-]+$/,
      "Group name can only contain letters, numbers, hyphens (-), and underscores (_). No spaces or special characters."
    ),
  description: z
    .string()
    .max(500, "Description is too long")
    .optional()
    .transform((value) => value?.trim() || undefined),
  isPublic: z.boolean(),
});

type GroupFormValues = z.infer<typeof groupSchema>;

const defaultValues: GroupFormValues = {
  institutionId: "",
  name: "",
  description: undefined,
  isPublic: true,
};

export const GroupCreateForm: FC<GroupCreateFormProps> = ({
  isOpen,
  onClose,
  userInstitutions,
}) => {
  const createGroupMutation = useCreateGroup();

  const {
    form,
    error: submitError,
    isSubmitting,
    handleSubmit,
  } = useFormWithMutation({
    schema: groupSchema,
    mutation: createGroupMutation,
    defaultValues,
    onSuccess: () => {
      onClose();
    },
    errorMessage: "Failed to create group. Please try again.",
  });

  useEffect(() => {
    if (isOpen && userInstitutions.length > 0) {
      const defaultInstitutionId = userInstitutions[0]?.id ?? "";
      if (!form.getValues("institutionId")) {
        form.setValue("institutionId", defaultInstitutionId);
      }
    }
    if (!isOpen) {
      form.reset(defaultValues);
    }
  }, [isOpen, userInstitutions, form]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-lg bg-white shadow-xl">
        <div className="border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Users className="h-5 w-5 text-blue-600" />
              <h2 className="text-lg font-semibold text-gray-900">
                Create New Group
              </h2>
            </div>
            <button
              onClick={() => {
                form.reset(defaultValues);
                onClose();
              }}
              className="cursor-pointer rounded-lg p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        <Form {...form}>
          <form onSubmit={handleSubmit} className="p-6">
            {userInstitutions.length === 0 ? (
              <div className="rounded-md bg-yellow-50 p-4 text-center">
                <Building className="mx-auto h-8 w-8 text-yellow-600" />
                <p className="mt-2 text-sm text-yellow-800">
                  You must be a member of an institution to create groups.
                </p>
                <p className="text-xs text-yellow-600">
                  Join an institution first, then create groups within it.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <FormField
                  control={form.control}
                  name="institutionId"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>
                        Institution <span className="text-red-500">*</span>
                      </FormLabel>
                      <Select
                        value={field.value}
                        onValueChange={field.onChange}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select an institution..." />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {userInstitutions.map((institution) => (
                            <SelectItem
                              key={institution.id}
                              value={institution.id}
                            >
                              {institution.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormDescription>
                        Groups belong to institutions and can be tagged as @
                        {"{group-name}"}
                      </FormDescription>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>
                        Group Name <span className="text-red-500">*</span>
                      </FormLabel>
                      <FormControl>
                        <Input
                          {...field}
                          placeholder="e.g., ai-research, book-club, project-team"
                          maxLength={100}
                        />
                      </FormControl>
                      <FormDescription>
                        Use letters, numbers, hyphens, and underscores only.
                        Perfect for @-tagging.
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="description"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Description</FormLabel>
                      <FormControl>
                        <Textarea
                          {...field}
                          value={field.value ?? ""}
                          onChange={(event) =>
                            field.onChange(event.target.value)
                          }
                          rows={3}
                          placeholder="What is this group about?"
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="isPublic"
                  render={({ field }) => (
                    <FormItem className="space-y-3">
                      <FormLabel>Privacy Setting</FormLabel>
                      <div className="space-y-2">
                        <label
                          className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition-colors ${
                            field.value
                              ? "border-blue-500 bg-blue-50"
                              : "border-gray-200 hover:border-gray-300"
                          }`}
                        >
                          <Checkbox
                            checked={field.value}
                            onCheckedChange={(checked) =>
                              field.onChange(Boolean(checked))
                            }
                          />
                          <Globe className="h-4 w-4 text-green-600" />
                          <div>
                            <div className="font-medium text-gray-900">
                              Public
                            </div>
                            <div className="text-xs text-gray-500">
                              Institution members can discover and join
                            </div>
                          </div>
                        </label>
                        <label
                          className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition-colors ${
                            !field.value
                              ? "border-blue-500 bg-blue-50"
                              : "border-gray-200 hover:border-gray-300"
                          }`}
                        >
                          <Checkbox
                            checked={!field.value}
                            onCheckedChange={(checked) =>
                              field.onChange(!checked)
                            }
                          />
                          <Lock className="h-4 w-4 text-orange-600" />
                          <div>
                            <div className="font-medium text-gray-900">
                              Private
                            </div>
                            <div className="text-xs text-gray-500">
                              Only visible to current members
                            </div>
                          </div>
                        </label>
                      </div>
                    </FormItem>
                  )}
                />

                {submitError && (
                  <ErrorAlert message={submitError} className="mb-4" />
                )}

                <div className="mt-6 flex justify-end gap-3">
                  <Button type="button" variant="outline" onClick={onClose}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={isSubmitting}>
                    {isSubmitting ? "Creating..." : "Create Group"}
                  </Button>
                </div>
              </div>
            )}
          </form>
        </Form>
      </div>
    </div>
  );
};
