import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { auth, isSignedIn } from "@/lib/auth";
import { emailService } from "@/lib/external-notifications/email";

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
    console.error("Error checking email configuration:", error);
    return NextResponse.json(
      {
        error: "Failed to check email configuration",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

// POST /api/email-test - Send test email
export async function POST(request: NextRequest) {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
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
        return NextResponse.json(
          { error: "No email address available for test" },
          { status: 400 }
        );
      }

      // Create a mock notification for testing
      const mockNotification = {
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
          name: session.user.name,
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

    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    console.error("Error in email test:", error);

    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request data", details: error.errors },
        { status: 400 }
      );
    }

    return NextResponse.json(
      {
        error: "Failed to process email test",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
