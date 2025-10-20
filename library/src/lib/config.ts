import { z } from "zod";

/**
 * Configuration schema with validation
 *
 * This centralizes all environment variable access and provides:
 * - Type safety for all config values
 * - Runtime validation with helpful error messages
 * - Default values where appropriate
 * - Clear documentation of required vs optional settings
 */
const configSchema = z.object({
  // Node environment
  nodeEnv: z.enum(["development", "production", "test"]).default("development"),

  // Database
  database: z.object({
    url: z.string().min(1, "DATABASE_URL is required"),
  }),

  // Redis / BullMQ
  redis: z.object({
    host: z.string().default("localhost"),
    port: z.coerce.number().default(6379),
    password: z.string().optional(),
    tls: z.coerce.boolean().default(false),
  }),

  // Authentication
  auth: z.object({
    secret: z
      .string()
      .min(32, "NEXTAUTH_SECRET must be at least 32 characters"),
    url: z.string().url("NEXTAUTH_URL must be a valid URL"),
    allowedDomains: z
      .string()
      .transform((s) => s.split(",").map((d) => d.trim()))
      .default(""),
    adminEmails: z
      .string()
      .transform((s) => s.split(",").map((email) => email.trim()))
      .default(""),
  }),

  // Discord OAuth (optional)
  discord: z.object({
    clientId: z.string().optional(),
    clientSecret: z.string().optional(),
    redirectUri: z.string().url().optional(),
    botToken: z.string().optional(),
  }),

  // Google OAuth (optional)
  google: z.object({
    clientId: z.string().optional(),
    clientSecret: z.string().optional(),
  }),

  // Email / Notifications
  email: z.object({
    enabled: z.coerce.boolean().default(false),
    fromAddress: z.string().email().optional(),
    fromName: z.string().optional(),
  }),

  // AWS SES (optional)
  aws: z.object({
    region: z.string().default("us-east-1"),
    profile: z.string().optional(),
    sesAccessKey: z.string().optional(),
    sesSecretKey: z.string().optional(),
  }),

  // SMTP (optional fallback for email)
  smtp: z.object({
    host: z.string().optional(),
    port: z.coerce.number().optional(),
    user: z.string().optional(),
    password: z.string().optional(),
  }),

  // LLM Services
  llm: z.object({
    anthropicApiKey: z.string().optional(),
  }),

  // Feature Flags
  features: z.object({
    devMode: z.coerce.boolean().default(false),
  }),
});

export type Config = z.infer<typeof configSchema>;

/**
 * Load and validate configuration from environment variables
 */
function loadConfig(): Config {
  try {
    const rawConfig = {
      nodeEnv: process.env.NODE_ENV,

      database: {
        url: process.env.DATABASE_URL,
      },

      redis: {
        host: process.env.REDIS_HOST,
        port: process.env.REDIS_PORT,
        password: process.env.REDIS_PASSWORD,
        tls: process.env.REDIS_TLS,
      },

      auth: {
        secret: process.env.NEXTAUTH_SECRET,
        url: process.env.NEXTAUTH_URL,
        allowedDomains: process.env.ALLOWED_EMAIL_DOMAINS,
        adminEmails: process.env.ADMIN_EMAILS,
      },

      discord: {
        clientId: process.env.DISCORD_CLIENT_ID,
        clientSecret: process.env.DISCORD_CLIENT_SECRET,
        redirectUri: process.env.DISCORD_REDIRECT_URI,
        botToken: process.env.DISCORD_BOT_TOKEN,
      },

      google: {
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      },

      email: {
        enabled: process.env.ENABLE_EMAIL_NOTIFICATIONS,
        fromAddress: process.env.EMAIL_FROM_ADDRESS,
        fromName: process.env.EMAIL_FROM_NAME,
      },

      aws: {
        region: process.env.AWS_REGION,
        profile: process.env.AWS_PROFILE,
        sesAccessKey: process.env.AWS_SES_ACCESS_KEY_ID,
        sesSecretKey: process.env.AWS_SES_SECRET_ACCESS_KEY,
      },

      smtp: {
        host: process.env.SMTP_HOST,
        port: process.env.SMTP_PORT,
        user: process.env.SMTP_USER,
        password: process.env.SMTP_PASSWORD,
      },

      llm: {
        anthropicApiKey: process.env.ANTHROPIC_API_KEY,
      },

      features: {
        devMode: process.env.DEV_MODE,
      },
    };

    return configSchema.parse(rawConfig);
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error("❌ Configuration validation failed:");
      console.error("");

      error.errors.forEach((err) => {
        const path = err.path.join(".");
        console.error(`  • ${path}: ${err.message}`);
      });

      console.error("");
      console.error(
        "Please check your .env.local file and ensure all required variables are set."
      );
    } else {
      console.error("❌ Failed to load configuration:", error);
    }

    throw new Error("Configuration validation failed");
  }
}

// Singleton config instance
let configInstance: Config | null = null;

/**
 * Get the application configuration
 *
 * Loads and validates configuration on first access, then returns cached instance.
 * Throws if required configuration is missing or invalid.
 */
export function getConfig(): Config {
  if (!configInstance) {
    configInstance = loadConfig();
  }
  return configInstance;
}

// Export default config for convenience
export const config = getConfig();
