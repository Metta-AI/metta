"use client";

import { FC, useState } from "react";
import { useAction } from "next-safe-action/hooks";
import { X, Building2 } from "lucide-react";

import { createInstitutionAction } from "@/institutions/actions/createInstitutionAction";

interface InstitutionCreateFormProps {
  isOpen: boolean;
  onClose: () => void;
}

export const InstitutionCreateForm: FC<InstitutionCreateFormProps> = ({
  isOpen,
  onClose,
}) => {
  const [formData, setFormData] = useState({
    name: "",
    domain: "",
    description: "",
    website: "",
    location: "",
    type: "COMPANY",
  });
  const [error, setError] = useState<string | null>(null);

  const { execute, isExecuting } = useAction(createInstitutionAction, {
    onSuccess: () => {
      setFormData({
        name: "",
        domain: "",
        description: "",
        website: "",
        location: "",
        type: "COMPANY",
      });
      setError(null);
      onClose();
    },
    onError: (error) => {
      console.error("Error creating institution:", error);

      // Extract error message from the error object (similar to NewPostForm)
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
        "Failed to create institution. Please try again.";

      setError(errorMessage);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const formDataObj = new FormData();
    Object.entries(formData).forEach(([key, value]) => {
      if (value) formDataObj.append(key, value);
    });

    execute(formDataObj);
  };

  const handleInputChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
    >
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));

    // Clear error when user starts typing in the name field
    if (name === "name" && error) {
      setError(null);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-lg bg-white p-6 shadow-xl">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Building2 className="h-5 w-5 text-blue-600" />
            <h2 className="text-lg font-semibold text-gray-900">
              Create Institution
            </h2>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Institution Name *
            </label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              required
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
              placeholder="e.g., Softmax Research"
            />
          </div>

          {/* Domain */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Domain
            </label>
            <input
              type="text"
              name="domain"
              value={formData.domain}
              onChange={handleInputChange}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
              placeholder="e.g., softmax.com"
            />
            <p className="mt-1 text-xs text-gray-500">
              Used for @-tagging (e.g., @softmax.com/team)
            </p>
          </div>

          {/* Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Institution Type *
            </label>
            <select
              name="type"
              value={formData.type}
              onChange={handleInputChange}
              required
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
            >
              <option value="UNIVERSITY">University</option>
              <option value="COMPANY">Company</option>
              <option value="RESEARCH_LAB">Research Lab</option>
              <option value="NONPROFIT">Nonprofit</option>
              <option value="GOVERNMENT">Government</option>
              <option value="OTHER">Other</option>
            </select>
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Description
            </label>
            <textarea
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              rows={3}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
              placeholder="Brief description of the institution..."
            />
          </div>

          {/* Website */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Website
            </label>
            <input
              type="url"
              name="website"
              value={formData.website}
              onChange={handleInputChange}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
              placeholder="https://..."
            />
          </div>

          {/* Location */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Location
            </label>
            <input
              type="text"
              name="location"
              value={formData.location}
              onChange={handleInputChange}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
              placeholder="e.g., San Francisco, CA"
            />
          </div>

          {/* Error Display */}
          {error && (
            <div className="rounded-md bg-red-50 p-3">
              <div className="text-sm text-red-700">{error}</div>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!formData.name || isExecuting}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
            >
              {isExecuting ? "Creating..." : "Create Institution"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
