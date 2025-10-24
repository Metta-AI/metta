"use client";

import {
  useForm,
  UseFormProps,
  UseFormReturn,
  FieldValues,
} from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { UseMutationResult } from "@tanstack/react-query";
import { z } from "zod";
import { useErrorHandling } from "./useErrorHandling";

interface UseFormWithMutationOptions<
  TSchema extends z.ZodType<any, any, any>,
  TData = unknown,
  TError = Error,
  TVariables = z.infer<TSchema>,
> {
  schema: TSchema;
  mutation: UseMutationResult<TData, TError, TVariables>;
  defaultValues?: UseFormProps<z.infer<TSchema>>["defaultValues"];
  onSuccess?: (data: TData) => void;
  onError?: (error: TError) => void;
  errorMessage?: string;
}

interface UseFormWithMutationReturn<TSchema extends z.ZodType<any, any, any>> {
  form: UseFormReturn<z.infer<TSchema>>;
  error: string | null;
  clearError: () => void;
  isSubmitting: boolean;
  handleSubmit: (e?: React.BaseSyntheticEvent) => Promise<void>;
}

/**
 * Consolidated form handling with mutation integration
 * Combines react-hook-form, zod validation, and React Query mutations
 */
export function useFormWithMutation<
  TSchema extends z.ZodType<any, any, any>,
  TData = unknown,
  TError = Error,
  TVariables = z.infer<TSchema>,
>({
  schema,
  mutation,
  defaultValues,
  onSuccess,
  onError,
  errorMessage = "An error occurred. Please try again.",
}: UseFormWithMutationOptions<
  TSchema,
  TData,
  TError,
  TVariables
>): UseFormWithMutationReturn<TSchema> {
  const form = useForm<z.infer<TSchema>>({
    resolver: zodResolver(schema),
    defaultValues,
  });

  const { error, setError, clearError } = useErrorHandling({
    fallbackMessage: errorMessage,
  });

  const onSubmit = (data: z.infer<TSchema>) => {
    clearError();
    mutation.mutate(data as TVariables, {
      onSuccess: (result) => {
        form.reset(defaultValues);
        clearError();
        onSuccess?.(result);
      },
      onError: (err) => {
        setError(err as Error);
        onError?.(err);
      },
    });
  };

  return {
    form,
    error,
    clearError,
    isSubmitting: mutation.isPending,
    handleSubmit: form.handleSubmit(onSubmit),
  };
}
