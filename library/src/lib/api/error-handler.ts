/**
 * Centralized API Error Handler
 *
 * Provides consistent error responses across all API routes with:
 * - Proper HTTP status codes
 * - Structured error responses
 * - Zod validation error formatting
 * - Development vs production error details
 */

import { NextResponse } from "next/server";
import { z } from "zod";
import { AppError, isAppError } from "../errors";
import { config } from "../config";
import { Logger } from "../logging/logger";

/**
 * Standard error response format
 */
export interface ErrorResponse {
  error: string;
  code: string;
  details?: unknown;
  timestamp?: string;
}

/**
 * Handle API errors and return appropriate NextResponse
 *
 * @param error - The error to handle
 * @param context - Optional context for logging (e.g., endpoint, user id)
 * @returns NextResponse with appropriate error status and message
 */
export function handleApiError(
  error: unknown,
  context?: Record<string, unknown>
): NextResponse<ErrorResponse> {
  // Log the error
  Logger.error("API Error", error, context);

  // Handle AppError instances
  if (isAppError(error)) {
    return NextResponse.json(
      {
        error: error.message,
        code: error.code,
        ...(error.details ? { details: error.details } : {}),
        timestamp: new Date().toISOString(),
      },
      { status: error.statusCode }
    );
  }

  // Handle Zod validation errors
  if (error instanceof z.ZodError) {
    const formattedErrors = error.errors.map((err) => ({
      path: err.path.join("."),
      message: err.message,
    }));

    return NextResponse.json(
      {
        error: "Validation failed",
        code: "VALIDATION_ERROR",
        details: formattedErrors,
        timestamp: new Date().toISOString(),
      },
      { status: 400 }
    );
  }

  // Handle standard Error instances
  if (error instanceof Error) {
    // In production, don't expose internal error details
    const message =
      config.nodeEnv === "production"
        ? "An unexpected error occurred"
        : error.message;

    return NextResponse.json(
      {
        error: message,
        code: "INTERNAL_ERROR",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }

  // Handle unknown error types
  return NextResponse.json(
    {
      error: "An unexpected error occurred",
      code: "INTERNAL_ERROR",
      timestamp: new Date().toISOString(),
    },
    { status: 500 }
  );
}

/**
 * Wrap an async API handler with error handling
 *
 * @param handler - The async handler function
 * @returns Wrapped handler with automatic error handling
 *
 * @example
 * export const GET = withErrorHandler(async (request) => {
 *   const data = await fetchData();
 *   return NextResponse.json(data);
 * });
 */
export function withErrorHandler<T extends any[], R>(
  handler: (...args: T) => Promise<NextResponse<R>>
) {
  return async (...args: T): Promise<NextResponse<R | ErrorResponse>> => {
    try {
      return await handler(...args);
    } catch (error) {
      return handleApiError(error);
    }
  };
}
