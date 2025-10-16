/**
 * Structured Logging Service
 *
 * Provides consistent logging across the application with:
 * - Structured log entries
 * - Different log levels (error, warn, info, debug)
 * - Development vs production formatting
 * - Optional integration with external logging services
 */

import { config } from "../config";

export type LogLevel = "error" | "warn" | "info" | "debug";

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
  context?: Record<string, unknown>;
}

/**
 * Logger class for structured logging
 */
export class Logger {
  /**
   * Log an error
   */
  static error(
    message: string,
    error: unknown,
    context?: Record<string, unknown>
  ): void {
    const logEntry = this.createLogEntry("error", message, error, context);

    if (config.nodeEnv === "development") {
      // Pretty print in development
      console.error(`❌ ${message}`);
      if (error instanceof Error) {
        console.error(error);
      }
      if (context) {
        console.error("Context:", context);
      }
    } else {
      // JSON format for production (easier to parse by log aggregators)
      console.error(JSON.stringify(logEntry));
    }

    // TODO: Send to external error tracking service (e.g., Sentry, DataDog)
    // this.sendToErrorTracking(logEntry);
  }

  /**
   * Log a warning
   */
  static warn(message: string, context?: Record<string, unknown>): void {
    const logEntry = this.createLogEntry("warn", message, undefined, context);

    if (config.nodeEnv === "development") {
      console.warn(`⚠️  ${message}`);
      if (context) {
        console.warn("Context:", context);
      }
    } else {
      console.warn(JSON.stringify(logEntry));
    }
  }

  /**
   * Log an info message
   */
  static info(message: string, context?: Record<string, unknown>): void {
    const logEntry = this.createLogEntry("info", message, undefined, context);

    if (config.nodeEnv === "development") {
      console.log(`ℹ️  ${message}`);
      if (context) {
        console.log("Context:", context);
      }
    } else {
      console.log(JSON.stringify(logEntry));
    }
  }

  /**
   * Log a debug message (only in development)
   */
  static debug(message: string, context?: Record<string, unknown>): void {
    if (config.nodeEnv !== "development") {
      return;
    }

    const logEntry = this.createLogEntry("debug", message, undefined, context);
    console.debug(`🔍 ${message}`);
    if (context) {
      console.debug("Context:", context);
    }
  }

  /**
   * Create a structured log entry
   */
  private static createLogEntry(
    level: LogLevel,
    message: string,
    error?: unknown,
    context?: Record<string, unknown>
  ): LogEntry {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
    };

    if (error instanceof Error) {
      entry.error = {
        name: error.name,
        message: error.message,
        stack: error.stack,
      };
    } else if (error) {
      entry.error = {
        name: "Unknown",
        message: String(error),
      };
    }

    if (context) {
      entry.context = context;
    }

    return entry;
  }
}
