/**
 * Application Error Hierarchy
 *
 * Provides consistent error handling across the application with:
 * - Type-safe error codes
 * - HTTP status code mapping
 * - Optional error details for debugging
 * - Better error messages for users
 */

/**
 * Base application error class
 *
 * All custom errors should extend this class
 */
export class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 500,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = this.constructor.name;

    // Maintains proper stack trace for where error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Convert error to a JSON-serializable object
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      statusCode: this.statusCode,
      ...(this.details && { details: this.details }),
    };
  }
}

/**
 * Validation error (400)
 *
 * Used when request data fails validation
 */
export class ValidationError extends AppError {
  constructor(message: string, details?: unknown) {
    super(message, "VALIDATION_ERROR", 400, details);
  }
}

/**
 * Not found error (404)
 *
 * Used when a requested resource doesn't exist
 */
export class NotFoundError extends AppError {
  constructor(resource: string, id?: string) {
    const message = id
      ? `${resource} with id '${id}' not found`
      : `${resource} not found`;
    super(message, "NOT_FOUND", 404);
  }
}

/**
 * Authorization error (403)
 *
 * Used when user lacks permission for an action
 */
export class AuthorizationError extends AppError {
  constructor(
    message: string = "You do not have permission to perform this action"
  ) {
    super(message, "FORBIDDEN", 403);
  }
}

/**
 * Authentication error (401)
 *
 * Used when user is not authenticated
 */
export class AuthenticationError extends AppError {
  constructor(message: string = "Authentication required") {
    super(message, "UNAUTHORIZED", 401);
  }
}

/**
 * Conflict error (409)
 *
 * Used when request conflicts with current state (e.g., duplicate resource)
 */
export class ConflictError extends AppError {
  constructor(message: string, details?: unknown) {
    super(message, "CONFLICT", 409, details);
  }
}

/**
 * Rate limit error (429)
 *
 * Used when user has exceeded rate limits
 */
export class RateLimitError extends AppError {
  constructor(
    message: string = "Too many requests. Please try again later.",
    retryAfter?: number
  ) {
    super(message, "RATE_LIMIT_EXCEEDED", 429, { retryAfter });
  }
}

/**
 * Bad request error (400)
 *
 * Used for general bad request scenarios
 */
export class BadRequestError extends AppError {
  constructor(message: string, details?: unknown) {
    super(message, "BAD_REQUEST", 400, details);
  }
}

/**
 * Service unavailable error (503)
 *
 * Used when external service is unavailable
 */
export class ServiceUnavailableError extends AppError {
  constructor(
    service: string,
    message: string = `${service} is temporarily unavailable`
  ) {
    super(message, "SERVICE_UNAVAILABLE", 503, { service });
  }
}

/**
 * Internal server error (500)
 *
 * Used for unexpected errors
 */
export class InternalServerError extends AppError {
  constructor(
    message: string = "An unexpected error occurred",
    details?: unknown
  ) {
    super(message, "INTERNAL_ERROR", 500, details);
  }
}

/**
 * Check if an error is an instance of AppError
 */
export function isAppError(error: unknown): error is AppError {
  return error instanceof AppError;
}

/**
 * Get a user-friendly error message
 *
 * Sanitizes error messages to avoid leaking sensitive information
 */
export function getUserMessage(error: unknown): string {
  if (isAppError(error)) {
    return error.message;
  }

  if (error instanceof Error) {
    // Don't expose internal error messages in production
    if (process.env.NODE_ENV === "production") {
      return "An unexpected error occurred";
    }
    return error.message;
  }

  return "An unexpected error occurred";
}

/**
 * Get HTTP status code from error
 */
export function getStatusCode(error: unknown): number {
  if (isAppError(error)) {
    return error.statusCode;
  }

  // Default to 500 for unknown errors
  return 500;
}
