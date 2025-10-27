/**
 * S3 Storage Service
 *
 * Handles file uploads to AWS S3 with proper configuration,
 * error handling, and support for both AWS profile and explicit credentials.
 */

import {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
  GetObjectCommand,
} from "@aws-sdk/client-s3";
import { fromIni } from "@aws-sdk/credential-providers";
import { config } from "./config";
import { Logger } from "./logging/logger";
import { randomUUID } from "crypto";

export interface UploadResult {
  key: string;
  url: string;
  bucket: string;
  size: number;
}

export class S3Service {
  private client?: S3Client;
  private bucketName: string;
  private region: string;
  private isConfigured: boolean;

  constructor() {
    this.bucketName = config.s3.bucketName || "";
    this.region = config.s3.region;
    this.isConfigured = false;

    if (!this.bucketName) {
      Logger.info("S3 service NOT configured: Missing AWS_S3_BUCKET");
      return;
    }

    try {
      // Try explicit credentials first (like SES does)
      if (config.aws.s3AccessKey && config.aws.s3SecretKey) {
        this.client = new S3Client({
          region: this.region,
          credentials: {
            accessKeyId: config.aws.s3AccessKey,
            secretAccessKey: config.aws.s3SecretKey,
          },
        });
        this.isConfigured = true;
        Logger.info("S3 service configured with explicit credentials", {
          bucket: this.bucketName,
          region: this.region,
        });
      }
      // Fall back to AWS profile (for dev)
      else if (config.aws.profile) {
        this.client = new S3Client({
          region: this.region,
          credentials: fromIni({ profile: config.aws.profile }),
        });
        this.isConfigured = true;
        Logger.info("S3 service configured with AWS profile", {
          bucket: this.bucketName,
          region: this.region,
          profile: config.aws.profile,
        });
      }
      // Fall back to default credential chain (for production IAM roles)
      else {
        this.client = new S3Client({
          region: this.region,
        });
        this.isConfigured = true;
        Logger.info("S3 service configured with default credential chain", {
          bucket: this.bucketName,
          region: this.region,
        });
      }
    } catch (error) {
      Logger.error("Failed to initialize S3 client", error);
      this.isConfigured = false;
    }
  }

  /**
   * Upload a file to S3
   * @param buffer - File buffer
   * @param contentType - MIME type
   * @param folder - Optional folder prefix (e.g., 'images', 'uploads')
   * @param filename - Optional custom filename (otherwise generates UUID)
   */
  async uploadFile(
    buffer: Buffer,
    contentType: string,
    folder: string = "uploads",
    filename?: string
  ): Promise<UploadResult> {
    if (!this.isConfigured || !this.client) {
      throw new Error(
        "S3 service not configured. Please set AWS_S3_BUCKET and AWS credentials."
      );
    }

    // Generate a unique key
    const timestamp = Date.now();
    const uuid = randomUUID();
    const extension = this.getExtensionFromMimeType(contentType);
    const key = filename
      ? `${folder}/${timestamp}-${filename}`
      : `${folder}/${timestamp}-${uuid}${extension}`;

    try {
      Logger.debug("Uploading file to S3", {
        bucket: this.bucketName,
        key,
        size: buffer.length,
        contentType,
      });

      const command = new PutObjectCommand({
        Bucket: this.bucketName,
        Key: key,
        Body: buffer,
        ContentType: contentType,
        // Note: ACL is not set - bucket policy should handle public access
        // or use pre-signed URLs for access control
      });

      await this.client.send(command);

      // Construct the public URL
      const url = `https://${this.bucketName}.s3.${this.region}.amazonaws.com/${key}`;

      Logger.info("File uploaded successfully to S3", {
        key,
        url,
        size: buffer.length,
      });

      return {
        key,
        url,
        bucket: this.bucketName,
        size: buffer.length,
      };
    } catch (error) {
      Logger.error("Failed to upload file to S3", error, {
        bucket: this.bucketName,
        key,
      });
      throw new Error(
        `S3 upload failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }

  /**
   * Delete a file from S3
   * @param key - The S3 object key
   */
  async deleteFile(key: string): Promise<boolean> {
    if (!this.isConfigured || !this.client) {
      Logger.warn("S3 service not configured, cannot delete file");
      return false;
    }

    try {
      Logger.debug("Deleting file from S3", {
        bucket: this.bucketName,
        key,
      });

      const command = new DeleteObjectCommand({
        Bucket: this.bucketName,
        Key: key,
      });

      await this.client.send(command);

      Logger.info("File deleted successfully from S3", { key });
      return true;
    } catch (error) {
      Logger.error("Failed to delete file from S3", error, {
        bucket: this.bucketName,
        key,
      });
      return false;
    }
  }

  /**
   * Check if S3 service is properly configured
   */
  isReady(): boolean {
    return this.isConfigured;
  }

  /**
   * Get configuration info for debugging
   */
  getConfigurationInfo(): {
    configured: boolean;
    bucket: string | null;
    region: string;
  } {
    return {
      configured: this.isConfigured,
      bucket: this.bucketName || null,
      region: this.region,
    };
  }

  /**
   * Helper to get file extension from MIME type
   */
  private getExtensionFromMimeType(mimeType: string): string {
    const extensions: Record<string, string> = {
      "image/jpeg": ".jpg",
      "image/jpg": ".jpg",
      "image/png": ".png",
      "image/gif": ".gif",
      "image/webp": ".webp",
      "application/pdf": ".pdf",
    };
    return extensions[mimeType] || "";
  }

  /**
   * Test S3 connection and permissions
   */
  async testConfiguration(): Promise<boolean> {
    if (!this.isConfigured || !this.client) {
      Logger.warn("S3 not configured");
      return false;
    }

    try {
      // Try a simple upload and delete test
      const testKey = `test/${Date.now()}-test.txt`;
      const testBuffer = Buffer.from("S3 configuration test");

      await this.client.send(
        new PutObjectCommand({
          Bucket: this.bucketName,
          Key: testKey,
          Body: testBuffer,
          ContentType: "text/plain",
        })
      );

      await this.client.send(
        new DeleteObjectCommand({
          Bucket: this.bucketName,
          Key: testKey,
        })
      );

      Logger.info("✅ S3 configuration test successful");
      return true;
    } catch (error) {
      Logger.error("❌ S3 configuration test failed", error);
      return false;
    }
  }
}

// Export singleton instance
export const s3Service = new S3Service();
