type ServerError = {
  message?: string;
};

type ValidationError = {
  message?: string;
};

type ActionErrorShape = {
  error?: {
    serverError?: unknown;
    validationErrors?: unknown;
  };
};

export function extractActionErrorMessage(
  error: unknown,
  fallbackMessage: string
): string {
  if (!error || typeof error !== "object") {
    return fallbackMessage;
  }

  const errorContainer = error as ActionErrorShape;
  const serverError = errorContainer.error?.serverError;
  const validationErrors = errorContainer.error?.validationErrors;

  if (typeof serverError === "string" && serverError.trim()) {
    return serverError;
  }

  if (
    serverError &&
    typeof serverError === "object" &&
    "message" in serverError
  ) {
    const message = (serverError as ServerError).message;
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }

  if (Array.isArray(validationErrors)) {
    for (const validationError of validationErrors) {
      if (
        validationError &&
        typeof validationError === "object" &&
        "message" in validationError
      ) {
        const message = (validationError as ValidationError).message;
        if (typeof message === "string" && message.trim()) {
          return message;
        }
      }
    }
  }

  return fallbackMessage;
}
