"use client";

import { FC, useEffect } from "react";
import { z } from "zod";
import { X, Building2 } from "lucide-react";

import { useFormWithMutation } from "@/lib/hooks/useFormWithMutation";
import { useCreateInstitution } from "@/hooks/mutations/admin";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";

interface InstitutionCreateFormProps {
  isOpen: boolean;
  onClose: () => void;
}

const institutionSchema = z.object({
  name: z.string().min(1, "Institution name is required"),
  domain: z
    .string()
    .max(255, "Domain is too long")
    .optional()
    .transform((value) => value?.trim() || undefined),
  description: z
    .string()
    .max(500, "Description is too long")
    .optional()
    .transform((value) => value?.trim() || undefined),
  website: z
    .string()
    .url("Please enter a valid URL")
    .optional()
    .or(z.literal(""))
    .transform((value) => value || undefined),
  location: z
    .string()
    .max(255, "Location is too long")
    .optional()
    .transform((value) => value?.trim() || undefined),
  type: z.enum([
    "UNIVERSITY",
    "COMPANY",
    "RESEARCH_LAB",
    "NONPROFIT",
    "GOVERNMENT",
    "OTHER",
  ]),
});

type InstitutionFormValues = z.infer<typeof institutionSchema>;

const defaultValues: InstitutionFormValues = {
  name: "",
  domain: undefined,
  description: undefined,
  website: undefined,
  location: undefined,
  type: "COMPANY",
};

export const InstitutionCreateForm: FC<InstitutionCreateFormProps> = ({
  isOpen,
  onClose,
}) => {
  const createInstitutionMutation = useCreateInstitution();

  const {
    form,
    error: submitError,
    isSubmitting,
    handleSubmit,
  } = useFormWithMutation({
    schema: institutionSchema,
    mutation: createInstitutionMutation,
    defaultValues,
    onSuccess: () => {
      onClose();
    },
    errorMessage: "Failed to create institution. Please try again.",
  });

  useEffect(() => {
    if (!isOpen) {
      form.reset(defaultValues);
    }
  }, [isOpen, form]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-lg bg-white p-6 shadow-xl">
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

        <Form {...form}>
          <form onSubmit={handleSubmit} className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Institution Name *</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      placeholder="e.g., Softmax Research"
                      autoComplete="organization"
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="domain"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Domain</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      value={field.value ?? ""}
                      onChange={(event) => field.onChange(event.target.value)}
                      placeholder="e.g., softmax.com"
                    />
                  </FormControl>
                  <FormDescription>
                    Used for @-tagging (e.g., @softmax.com/team)
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Institution Type *</FormLabel>
                  <Select value={field.value} onValueChange={field.onChange}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="UNIVERSITY">University</SelectItem>
                      <SelectItem value="COMPANY">Company</SelectItem>
                      <SelectItem value="RESEARCH_LAB">Research Lab</SelectItem>
                      <SelectItem value="NONPROFIT">Nonprofit</SelectItem>
                      <SelectItem value="GOVERNMENT">Government</SelectItem>
                      <SelectItem value="OTHER">Other</SelectItem>
                    </SelectContent>
                  </Select>
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
                      onChange={(event) => field.onChange(event.target.value)}
                      rows={3}
                      placeholder="Brief description of the institution..."
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="website"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Website</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      value={field.value ?? ""}
                      onChange={(event) => field.onChange(event.target.value)}
                      placeholder="https://..."
                      type="url"
                      autoComplete="url"
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="location"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Location</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      value={field.value ?? ""}
                      onChange={(event) => field.onChange(event.target.value)}
                      placeholder="e.g., San Francisco, CA"
                      autoComplete="address-level2"
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            {submitError && (
              <ErrorAlert message={submitError} className="mb-4" />
            )}

            <div className="flex justify-end gap-3 pt-4">
              <Button type="button" variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Creating..." : "Create Institution"}
              </Button>
            </div>
          </form>
        </Form>
      </div>
    </div>
  );
};
