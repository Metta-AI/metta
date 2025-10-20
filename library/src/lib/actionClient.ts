import {
  createSafeActionClient,
  DEFAULT_SERVER_ERROR_MESSAGE,
} from "next-safe-action";
import { isAppError, getUserMessage } from "./errors";
import { Logger } from "./logging/logger";

export const actionClient = createSafeActionClient({
  handleServerError(e) {
    // Log the error
    Logger.error("Server Action Error", e);

    // Handle our custom AppError instances
    if (isAppError(e)) {
      return e.message;
    }

    // For unexpected errors, return generic message in production
    if (process.env.NODE_ENV === "production") {
      return DEFAULT_SERVER_ERROR_MESSAGE;
    }

    // In development, show actual error message
    return getUserMessage(e);
  },
});
