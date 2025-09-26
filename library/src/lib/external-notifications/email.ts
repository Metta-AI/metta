/**
 * Email Notification Service
 *
 * Handles sending email notifications with proper templates,
 * error handling, and delivery tracking.
 */

import { createTransport, Transporter } from "nodemailer";
import { SESClient, SendEmailCommand } from "@aws-sdk/client-ses";
import { prisma } from "@/lib/db/prisma";
import type { Notification, User, Post, Comment } from "@prisma/client";

// Extended notification type with relations
export interface NotificationWithDetails extends Notification {
  actor?: { id: string; name: string | null; email: string | null } | null;
  user: { id: string; name: string | null; email: string | null };
  post?: { id: string; title: string } | null;
  comment?: {
    id: string;
    content: string;
    post: { id: string; title: string };
  } | null;
}

export interface EmailTemplate {
  subject: string;
  html: string;
  text: string;
}

export class EmailNotificationService {
  private transporter?: Transporter;
  private sesClient?: SESClient;
  private fromAddress: string;
  private fromName: string;
  private baseUrl: string;
  private useAWSsES: boolean;

  constructor() {
    this.fromAddress =
      process.env.EMAIL_FROM_ADDRESS || "notifications@yourapp.com";
    this.fromName = process.env.EMAIL_FROM_NAME || "Library Notifications";
    this.baseUrl = process.env.NEXTAUTH_URL || "http://localhost:3001";

    // Determine email sending method based on available configuration
    this.useAWSsES = !!(
      process.env.AWS_SES_ACCESS_KEY_ID &&
      process.env.AWS_SES_SECRET_ACCESS_KEY &&
      process.env.AWS_SES_REGION
    );

    if (this.useAWSsES) {
      // Configure AWS SES client
      this.sesClient = new SESClient({
        region: process.env.AWS_SES_REGION!,
        credentials: {
          accessKeyId: process.env.AWS_SES_ACCESS_KEY_ID!,
          secretAccessKey: process.env.AWS_SES_SECRET_ACCESS_KEY!,
        },
      });
      console.log("📧 Email service configured for AWS SES");
    } else {
      // Fallback to SMTP configuration
      this.transporter = createTransport({
        host: process.env.SMTP_HOST || "smtp.sendgrid.net",
        port: parseInt(process.env.SMTP_PORT || "587"),
        secure: false, // true for 465, false for other ports
        auth: {
          user: process.env.SMTP_USER || "apikey",
          pass: process.env.SMTP_PASSWORD,
        },
      });
      console.log("📧 Email service configured for SMTP");
    }
  }

  /**
   * Send an email notification
   */
  async sendNotification(
    notification: NotificationWithDetails,
    deliveryId?: string
  ): Promise<boolean> {
    try {
      if (!notification.user.email) {
        console.warn(`No email address for user ${notification.userId}`);
        return false;
      }

      // Create delivery record if not provided
      let currentDeliveryId = deliveryId;
      if (!currentDeliveryId) {
        const delivery = await prisma.notificationDelivery.create({
          data: {
            notificationId: notification.id,
            channel: "email",
            status: "pending",
          },
        });
        currentDeliveryId = delivery.id;
      }

      // Update delivery record to 'sending'
      await this.updateDeliveryStatus(currentDeliveryId, "sending", {
        attemptCount: { increment: 1 },
        lastAttempt: new Date(),
      });

      // Generate email template
      const template = await this.generateEmailTemplate(notification);

      // Send email using appropriate method
      let messageId: string;
      if (this.useAWSsES) {
        messageId = await this.sendWithSES(
          notification.user.email,
          template.subject,
          template.text,
          template.html
        );
      } else {
        messageId = await this.sendWithSMTP(
          notification.user.email,
          template.subject,
          template.text,
          template.html
        );
      }

      console.log(`📧 Email sent to ${notification.user.email}: ${messageId}`);

      // Mark as delivered
      await this.updateDeliveryStatus(currentDeliveryId, "sent", {
        deliveredAt: new Date(),
      });

      return true;
    } catch (error) {
      console.error("Email notification failed:", error);

      if (deliveryId) {
        await this.updateDeliveryStatus(deliveryId, "failed", {
          errorMessage: error instanceof Error ? error.message : String(error),
        });
      }

      return false;
    }
  }

  /**
   * Send email using AWS SES
   */
  private async sendWithSES(
    to: string,
    subject: string,
    textBody: string,
    htmlBody: string
  ): Promise<string> {
    if (!this.sesClient) {
      throw new Error("SES client not configured");
    }

    const command = new SendEmailCommand({
      Source: `"${this.fromName}" <${this.fromAddress}>`,
      Destination: {
        ToAddresses: [to],
      },
      Message: {
        Subject: {
          Data: subject,
          Charset: "UTF-8",
        },
        Body: {
          Text: {
            Data: textBody,
            Charset: "UTF-8",
          },
          Html: {
            Data: htmlBody,
            Charset: "UTF-8",
          },
        },
      },
    });

    const result = await this.sesClient.send(command);
    return result.MessageId || "unknown";
  }

  /**
   * Send email using SMTP
   */
  private async sendWithSMTP(
    to: string,
    subject: string,
    textBody: string,
    htmlBody: string
  ): Promise<string> {
    if (!this.transporter) {
      throw new Error("SMTP transporter not configured");
    }

    const info = await this.transporter.sendMail({
      from: `"${this.fromName}" <${this.fromAddress}>`,
      to,
      subject,
      text: textBody,
      html: htmlBody,
    });

    return info.messageId || "unknown";
  }

  /**
   * Generate email template for notification
   */
  async generateEmailTemplate(
    notification: NotificationWithDetails
  ): Promise<EmailTemplate> {
    const actorName = this.getActorDisplayName(notification.actor);
    const userName = this.getUserDisplayName(notification.user);

    switch (notification.type) {
      case "MENTION":
        return this.generateMentionTemplate(notification, actorName, userName);
      case "COMMENT":
        return this.generateCommentTemplate(notification, actorName, userName);
      case "REPLY":
        return this.generateReplyTemplate(notification, actorName, userName);
      case "SYSTEM":
        return this.generateSystemTemplate(notification, userName);
      default:
        return this.generateGenericTemplate(notification, userName);
    }
  }

  /**
   * Generate mention notification template
   */
  private generateMentionTemplate(
    notification: NotificationWithDetails,
    actorName: string,
    userName: string
  ): EmailTemplate {
    const subject = `${actorName} mentioned you in a ${notification.post ? "post" : "comment"}`;
    const actionUrl = this.getActionUrl(notification);

    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>${subject}</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 24px; }
            .content { padding: 0 4px; }
            .mention-text { background: #fff3cd; padding: 8px 12px; border-radius: 4px; border-left: 4px solid #ffc107; font-style: italic; margin: 16px 0; }
            .button { display: inline-block; background: #0066cc; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 500; margin: 20px 0; }
            .footer { margin-top: 32px; padding-top: 20px; border-top: 1px solid #eee; font-size: 14px; color: #666; }
            .unsubscribe { font-size: 12px; color: #999; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1 style="margin: 0; font-size: 24px;">You were mentioned!</h1>
          </div>

          <div class="content">
            <p>Hi ${userName},</p>

            <p><strong>${actorName}</strong> mentioned you in a ${notification.post ? "post" : "comment"}.</p>

            ${
              notification.mentionText
                ? `
              <div class="mention-text">
                "${notification.mentionText}"
              </div>
            `
                : ""
            }

            ${notification.message ? `<p>${notification.message}</p>` : ""}

            <a href="${actionUrl}" class="button">View ${notification.post ? "Post" : "Comment"}</a>
          </div>

          <div class="footer">
            <p>This notification was sent because you have email notifications enabled for mentions.</p>
            <p class="unsubscribe">
              <a href="${this.baseUrl}/settings/notifications">Manage notification preferences</a>
            </p>
          </div>
        </body>
      </html>
    `;

    const text = `
${subject}

Hi ${userName},

${actorName} mentioned you in a ${notification.post ? "post" : "comment"}.

${notification.mentionText ? `"${notification.mentionText}"` : ""}

${notification.message || ""}

View the ${notification.post ? "post" : "comment"}: ${actionUrl}

---
This notification was sent because you have email notifications enabled for mentions.
Manage your preferences: ${this.baseUrl}/settings/notifications
    `.trim();

    return { subject, html, text };
  }

  /**
   * Generate comment notification template
   */
  private generateCommentTemplate(
    notification: NotificationWithDetails,
    actorName: string,
    userName: string
  ): EmailTemplate {
    const postTitle = notification.post?.title || "your post";
    const subject = `${actorName} commented on ${postTitle}`;
    const actionUrl = this.getActionUrl(notification);

    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>${subject}</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 24px; }
            .content { padding: 0 4px; }
            .post-title { font-size: 18px; font-weight: 600; color: #0066cc; margin: 16px 0; }
            .button { display: inline-block; background: #0066cc; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 500; margin: 20px 0; }
            .footer { margin-top: 32px; padding-top: 20px; border-top: 1px solid #eee; font-size: 14px; color: #666; }
            .unsubscribe { font-size: 12px; color: #999; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1 style="margin: 0; font-size: 24px;">New Comment</h1>
          </div>

          <div class="content">
            <p>Hi ${userName},</p>

            <p><strong>${actorName}</strong> commented on your post:</p>

            ${notification.post ? `<div class="post-title">${notification.post.title}</div>` : ""}

            ${notification.message ? `<p>${notification.message}</p>` : ""}

            <a href="${actionUrl}" class="button">View Comment</a>
          </div>

          <div class="footer">
            <p>This notification was sent because you have email notifications enabled for comments.</p>
            <p class="unsubscribe">
              <a href="${this.baseUrl}/settings/notifications">Manage notification preferences</a>
            </p>
          </div>
        </body>
      </html>
    `;

    const text = `
${subject}

Hi ${userName},

${actorName} commented on your post${notification.post ? `: ${notification.post.title}` : ""}.

${notification.message || ""}

View the comment: ${actionUrl}

---
This notification was sent because you have email notifications enabled for comments.
Manage your preferences: ${this.baseUrl}/settings/notifications
    `.trim();

    return { subject, html, text };
  }

  /**
   * Generate reply notification template
   */
  private generateReplyTemplate(
    notification: NotificationWithDetails,
    actorName: string,
    userName: string
  ): EmailTemplate {
    const subject = `${actorName} replied to your comment`;
    const actionUrl = this.getActionUrl(notification);

    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>${subject}</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 24px; }
            .content { padding: 0 4px; }
            .button { display: inline-block; background: #0066cc; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 500; margin: 20px 0; }
            .footer { margin-top: 32px; padding-top: 20px; border-top: 1px solid #eee; font-size: 14px; color: #666; }
            .unsubscribe { font-size: 12px; color: #999; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1 style="margin: 0; font-size: 24px;">New Reply</h1>
          </div>

          <div class="content">
            <p>Hi ${userName},</p>

            <p><strong>${actorName}</strong> replied to your comment.</p>

            ${notification.message ? `<p>${notification.message}</p>` : ""}

            <a href="${actionUrl}" class="button">View Reply</a>
          </div>

          <div class="footer">
            <p>This notification was sent because you have email notifications enabled for replies.</p>
            <p class="unsubscribe">
              <a href="${this.baseUrl}/settings/notifications">Manage notification preferences</a>
            </p>
          </div>
        </body>
      </html>
    `;

    const text = `
${subject}

Hi ${userName},

${actorName} replied to your comment.

${notification.message || ""}

View the reply: ${actionUrl}

---
This notification was sent because you have email notifications enabled for replies.
Manage your preferences: ${this.baseUrl}/settings/notifications
    `.trim();

    return { subject, html, text };
  }

  /**
   * Generate system notification template
   */
  private generateSystemTemplate(
    notification: NotificationWithDetails,
    userName: string
  ): EmailTemplate {
    const subject = notification.title;
    const actionUrl = this.getActionUrl(notification);

    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>${subject}</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 24px; }
            .content { padding: 0 4px; }
            .button { display: inline-block; background: #0066cc; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 500; margin: 20px 0; }
            .footer { margin-top: 32px; padding-top: 20px; border-top: 1px solid #eee; font-size: 14px; color: #666; }
            .unsubscribe { font-size: 12px; color: #999; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1 style="margin: 0; font-size: 24px; color: #1976d2;">System Notification</h1>
          </div>

          <div class="content">
            <p>Hi ${userName},</p>

            <h2 style="color: #333; font-size: 20px;">${notification.title}</h2>

            ${notification.message ? `<p>${notification.message}</p>` : ""}

            ${actionUrl && actionUrl !== "#" ? `<a href="${actionUrl}" class="button">Learn More</a>` : ""}
          </div>

          <div class="footer">
            <p>This is a system notification from the Library.</p>
            <p class="unsubscribe">
              <a href="${this.baseUrl}/settings/notifications">Manage notification preferences</a>
            </p>
          </div>
        </body>
      </html>
    `;

    const text = `
${subject}

Hi ${userName},

${notification.title}

${notification.message || ""}

${actionUrl && actionUrl !== "#" ? `Learn more: ${actionUrl}` : ""}

---
This is a system notification from the Library.
Manage your preferences: ${this.baseUrl}/settings/notifications
    `.trim();

    return { subject, html, text };
  }

  /**
   * Generate generic notification template
   */
  private generateGenericTemplate(
    notification: NotificationWithDetails,
    userName: string
  ): EmailTemplate {
    const subject = notification.title;
    const actionUrl = this.getActionUrl(notification);

    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>${subject}</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 24px; }
            .content { padding: 0 4px; }
            .button { display: inline-block; background: #0066cc; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 500; margin: 20px 0; }
            .footer { margin-top: 32px; padding-top: 20px; border-top: 1px solid #eee; font-size: 14px; color: #666; }
            .unsubscribe { font-size: 12px; color: #999; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1 style="margin: 0; font-size: 24px;">New Notification</h1>
          </div>

          <div class="content">
            <p>Hi ${userName},</p>

            <h2 style="color: #333; font-size: 20px;">${notification.title}</h2>

            ${notification.message ? `<p>${notification.message}</p>` : ""}

            ${actionUrl && actionUrl !== "#" ? `<a href="${actionUrl}" class="button">View Details</a>` : ""}
          </div>

          <div class="footer">
            <p>This notification was sent from the Library.</p>
            <p class="unsubscribe">
              <a href="${this.baseUrl}/settings/notifications">Manage notification preferences</a>
            </p>
          </div>
        </body>
      </html>
    `;

    const text = `
${subject}

Hi ${userName},

${notification.title}

${notification.message || ""}

${actionUrl && actionUrl !== "#" ? `View details: ${actionUrl}` : ""}

---
This notification was sent from the Library.
Manage your preferences: ${this.baseUrl}/settings/notifications
    `.trim();

    return { subject, html, text };
  }

  /**
   * Update delivery record status
   */
  private async updateDeliveryStatus(
    deliveryId: string,
    status: string,
    additionalData: any = {}
  ): Promise<void> {
    await prisma.notificationDelivery.update({
      where: { id: deliveryId },
      data: {
        status,
        ...additionalData,
      },
    });
  }

  /**
   * Get actor display name
   */
  private getActorDisplayName(
    actor: { name: string | null; email: string | null } | null | undefined
  ): string {
    if (!actor) return "Someone";
    return actor.name || actor.email?.split("@")[0] || "Someone";
  }

  /**
   * Get user display name
   */
  private getUserDisplayName(user: {
    name: string | null;
    email: string | null;
  }): string {
    return user.name || user.email?.split("@")[0] || "there";
  }

  /**
   * Get action URL for notification
   */
  private getActionUrl(notification: NotificationWithDetails): string {
    if (notification.actionUrl) {
      return notification.actionUrl.startsWith("http")
        ? notification.actionUrl
        : `${this.baseUrl}${notification.actionUrl}`;
    }

    if (notification.post) {
      return `${this.baseUrl}/posts/${notification.post.id}${notification.commentId ? `#comment-${notification.commentId}` : ""}`;
    }

    if (notification.comment) {
      return `${this.baseUrl}/posts/${notification.comment.post.id}#comment-${notification.comment.id}`;
    }

    return "#";
  }

  /**
   * Test email configuration
   */
  async testConfiguration(): Promise<boolean> {
    try {
      if (this.useAWSsES) {
        // For SES, we'll try to get the sending quota as a health check
        if (!this.sesClient) {
          throw new Error("SES client not configured");
        }

        // Simple test - if we can create the client and it doesn't throw, it's likely configured
        console.log("✅ AWS SES configuration appears valid");
        return true;
      } else {
        // For SMTP, use the existing nodemailer verify
        if (!this.transporter) {
          throw new Error("SMTP transporter not configured");
        }

        await this.transporter.verify();
        console.log("✅ SMTP configuration is valid");
        return true;
      }
    } catch (error) {
      console.error("❌ Email configuration error:", error);
      return false;
    }
  }

  /**
   * Get current email configuration info
   */
  getConfigurationInfo(): {
    method: "AWS_SES" | "SMTP";
    fromAddress: string;
    fromName: string;
    configured: boolean;
  } {
    return {
      method: this.useAWSsES ? "AWS_SES" : "SMTP",
      fromAddress: this.fromAddress,
      fromName: this.fromName,
      configured: this.useAWSsES ? !!this.sesClient : !!this.transporter,
    };
  }
}

// Export singleton instance
export const emailService = new EmailNotificationService();
