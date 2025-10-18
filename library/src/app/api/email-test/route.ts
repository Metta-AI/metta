import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { auth, isSignedIn } from "@/lib/auth";
import {
  emailService,
  type NotificationWithDetails,
} from "@/lib/external-notifications/email";
import { AuthenticationError, BadRequestError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";

// Schema for test email request
const testEmailSchema = z.object({
  action: z.enum(["config", "send"]),
  testEmail: z.string().email().optional(),
});

// GET /api/email-test - Check email configuration
export async function GET() {
  try {
    const config = emailService.getConfigurationInfo();
    const isValid = await emailService.testConfiguration();

    return NextResponse.json({
      configuration: config,
      isValid,
      message: isValid
        ? `✅ Email service configured for ${config.method}`
        : `❌ Email configuration invalid`,
    });
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/email-test" });
  }
}

// POST /api/email-test - Send test email
export async function POST(request: NextRequest) {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      throw new AuthenticationError();
    }

    const body = await request.json();
    const { action, testEmail } = testEmailSchema.parse(body);

    if (action === "config") {
      // Just return configuration info
      const config = emailService.getConfigurationInfo();
      const isValid = await emailService.testConfiguration();

      return NextResponse.json({
        configuration: config,
        isValid,
        message: isValid
          ? `✅ Email service configured for ${config.method}`
          : `❌ Email configuration invalid`,
      });
    }

    if (action === "send") {
      // Send test email
      const recipientEmail = testEmail || session.user.email;

      if (!recipientEmail) {
        throw new BadRequestError("No email address available for test");
      }

      // Create a mock notification for testing
      const mockNotification: NotificationWithDetails = {
        id: "test-notification",
        userId: session.user.id,
        type: "SYSTEM" as const,
        isRead: false,
        title: "Test Email Notification",
        message:
          "This is a test email to verify your notification system is working correctly.",
        actionUrl: null,
        createdAt: new Date(),
        updatedAt: new Date(),
        actorId: null,
        postId: null,
        commentId: null,
        mentionText: null,
        actor: null,
        user: {
          id: session.user.id,
          name: session.user.name ?? null,
          email: recipientEmail,
        },
        post: null,
        comment: null,
      };

      const success = await emailService.sendNotification(mockNotification);

      return NextResponse.json({
        success,
        recipient: recipientEmail,
        message: success
          ? `✅ Test email sent successfully to ${recipientEmail}`
          : `❌ Failed to send test email to ${recipientEmail}`,
      });
    }

    throw new BadRequestError("Invalid action");
  } catch (error) {
    return handleApiError(error, { endpoint: "POST /api/email-test" });
  }
}
