"use client";

import { FC, useState, useEffect } from "react";
import { useAction } from "next-safe-action/hooks";
import { X, Users, Globe, Lock, Building } from "lucide-react";

import { createGroupAction } from "@/groups/actions/createGroupAction";
import { loadUserInstitutions } from "@/posts/data/managed-institutions";

interface Institution {
  id: string;
  name: string;
}

interface GroupCreateFormProps {
  isOpen: boolean;
  onClose: () => void;
  userInstitutions: Institution[];
}

export const GroupCreateForm: FC<GroupCreateFormProps> = ({
  isOpen,
  onClose,
  userInstitutions,
}) => {
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    isPublic: true,
    institutionId: "",
  });

  // Auto-select first institution when dialog opens
  useEffect(() => {
    if (isOpen && userInstitutions.length > 0 && !formData.institutionId) {
      setFormData((prev) => ({
        ...prev,
        institutionId: userInstitutions[0].id,
      }));
    }
  }, [isOpen, userInstitutions, formData.institutionId]);

  const [nameError, setNameError] = useState<string | null>(null);

  const validateGroupName = (name: string) => {
    if (!name) {
      setNameError(null);
      return;
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      setNameError(
        "Group name can only contain letters, numbers, hyphens, and underscores (no spaces)"
      );
    } else {
      setNameError(null);
    }
  };

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newName = e.target.value;
    setFormData((prev) => ({ ...prev, name: newName }));
    validateGroupName(newName);
  };

  const { execute, isExecuting, result } = useAction(createGroupAction, {
    onSuccess: () => {
      setFormData({
        name: "",
        description: "",
        isPublic: true,
        institutionId: "",
      });
      setNameError(null);
      onClose();
    },
    onError: (error) => {
      console.error("Error creating group:", error);

      // Extract error message from the error object
      const serverError = error.error?.serverError;
      const validationErrors = error.error?.validationErrors;

      const errorMessage =
        (typeof serverError === "string" ? serverError : null) ||
        (typeof serverError === "object" &&
        serverError !== null &&
        "message" in serverError
          ? (serverError as any).message
          : null) ||
        (Array.isArray(validationErrors) &&
        validationErrors.length > 0 &&
        typeof validationErrors[0] === "object" &&
        validationErrors[0] !== null &&
        "message" in validationErrors[0]
          ? (validationErrors[0] as any).message
          : null) ||
        "Failed to create group. Please try again.";

      setNameError(errorMessage);
    },
  });

  const handleClose = () => {
    setFormData({
      name: "",
      description: "",
      isPublic: true,
      institutionId: "",
    });
    setNameError(null);
    onClose();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.institutionId) {
      return; // Should not happen if institutions are available
    }

    const formDataObj = new FormData();
    formDataObj.append("name", formData.name);
    formDataObj.append("description", formData.description);
    formDataObj.append("institutionId", formData.institutionId);
    if (formData.isPublic) {
      formDataObj.append("isPublic", "on");
    }

    execute(formDataObj);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-lg bg-white shadow-xl">
        {/* Header */}
        <div className="border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Users className="h-5 w-5 text-blue-600" />
              <h2 className="text-lg font-semibold text-gray-900">
                Create New Group
              </h2>
            </div>
            <button
              onClick={handleClose}
              className="rounded-lg p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Form */}
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
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Institution *
                </label>
                <select
                  value={formData.institutionId}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      institutionId: e.target.value,
                    }))
                  }
                  required
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                >
                  <option value="">Select an institution...</option>
                  {userInstitutions.map((institution) => (
                    <option key={institution.id} value={institution.id}>
                      {institution.name}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  Groups belong to institutions and can be tagged as @
                  {"{group-name}"}
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Group Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={handleNameChange}
                  required
                  maxLength={100}
                  className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm focus:ring-1 focus:outline-none ${
                    nameError
                      ? "border-red-300 focus:border-red-500 focus:ring-red-500"
                      : "border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  }`}
                  placeholder="e.g., ai-research, book-club, project-team"
                />
                {nameError && (
                  <p className="mt-1 text-xs text-red-600">{nameError}</p>
                )}
                <p className="mt-1 text-xs text-gray-500">
                  Use letters, numbers, hyphens, and underscores only. Perfect
                  for @-tagging!
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Description
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      description: e.target.value,
                    }))
                  }
                  rows={3}
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                  placeholder="What is this group about?"
                />
              </div>

              <div>
                <label className="mb-3 block text-sm font-medium text-gray-700">
                  Privacy Setting
                </label>
                <div className="space-y-2">
                  <div
                    className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition-colors ${
                      formData.isPublic
                        ? "border-blue-500 bg-blue-50"
                        : "border-gray-200 hover:border-gray-300"
                    }`}
                    onClick={() =>
                      setFormData((prev) => ({ ...prev, isPublic: true }))
                    }
                  >
                    <input
                      type="radio"
                      name="privacy"
                      checked={formData.isPublic}
                      onChange={() =>
                        setFormData((prev) => ({ ...prev, isPublic: true }))
                      }
                      className="text-blue-600"
                    />
                    <Globe className="h-4 w-4 text-green-600" />
                    <div>
                      <div className="font-medium text-gray-900">Public</div>
                      <div className="text-xs text-gray-500">
                        Institution members can discover and join
                      </div>
                    </div>
                  </div>

                  <div
                    className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition-colors ${
                      !formData.isPublic
                        ? "border-blue-500 bg-blue-50"
                        : "border-gray-200 hover:border-gray-300"
                    }`}
                    onClick={() =>
                      setFormData((prev) => ({ ...prev, isPublic: false }))
                    }
                  >
                    <input
                      type="radio"
                      name="privacy"
                      checked={!formData.isPublic}
                      onChange={() =>
                        setFormData((prev) => ({ ...prev, isPublic: false }))
                      }
                      className="text-blue-600"
                    />
                    <Lock className="h-4 w-4 text-orange-600" />
                    <div>
                      <div className="font-medium text-gray-900">Private</div>
                      <div className="text-xs text-gray-500">
                        Only visible to current members
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {result?.data?.message && (
                <div className="mt-4 rounded-md bg-green-50 p-3 text-sm text-green-800">
                  {result.data.message}
                </div>
              )}

              <div className="mt-6 flex justify-end gap-3">
                <button
                  type="button"
                  onClick={handleClose}
                  className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={
                    !formData.name.trim() ||
                    !formData.institutionId ||
                    !!nameError ||
                    isExecuting
                  }
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
                >
                  {isExecuting ? "Creating..." : "Create Group"}
                </button>
              </div>
            </div>
          )}
        </form>
      </div>
    </div>
  );
};
