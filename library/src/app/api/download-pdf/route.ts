import { NextRequest, NextResponse } from "next/server";
import { BadRequestError, ServiceUnavailableError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";
import { Logger } from "@/lib/logging/logger";

// Whitelist of allowed domains for PDF downloads
const ALLOWED_DOMAINS = [
  "arxiv.org",
  "www.arxiv.org",
  // Add other trusted academic/paper repositories as needed
];

// Maximum PDF size (50MB)
const MAX_PDF_SIZE = 50 * 1024 * 1024;

/**
 * Validate URL for security (prevent SSRF attacks)
 */
function validatePdfUrl(urlString: string): URL {
  let parsedUrl: URL;
  
  try {
    parsedUrl = new URL(urlString);
  } catch {
    throw new BadRequestError("Invalid URL format");
  }

  // Only allow HTTPS (no HTTP, file://, ftp://, etc.)
  if (parsedUrl.protocol !== "https:") {
    throw new BadRequestError("Only HTTPS URLs are allowed");
  }

  // Check domain whitelist
  const hostname = parsedUrl.hostname.toLowerCase();
  const isAllowed = ALLOWED_DOMAINS.some(
    (domain) => hostname === domain || hostname.endsWith(`.${domain}`)
  );

  if (!isAllowed) {
    throw new BadRequestError(
      `Domain not allowed. Only PDFs from: ${ALLOWED_DOMAINS.join(", ")}`
    );
  }

  // Block private IP ranges and localhost
  const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
  if (ipv4Regex.test(hostname)) {
    const octets = hostname.split(".").map(Number);
    
    // Block private ranges: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
    if (
      octets[0] === 10 ||
      (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31) ||
      (octets[0] === 192 && octets[1] === 168) ||
      octets[0] === 127 // localhost
    ) {
      throw new BadRequestError("Cannot access private IP addresses");
    }
  }

  // Block localhost by hostname
  if (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "::1"
  ) {
    throw new BadRequestError("Cannot access localhost");
  }

  return parsedUrl;
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const url = searchParams.get("url");
    const filename = searchParams.get("filename") || "document.pdf";

    if (!url) {
      throw new BadRequestError("URL parameter is required");
    }

    // Validate URL with security checks
    const validatedUrl = validatePdfUrl(url);

    // Log the request for security monitoring
    Logger.info("PDF download request", {
      url: validatedUrl.href,
      domain: validatedUrl.hostname,
      filename,
    });

    // Fetch the PDF from the validated URL with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    let pdfBuffer: ArrayBuffer;

    try {
      const response = await fetch(validatedUrl.href, {
        headers: {
          "User-Agent": "Mozilla/5.0 (compatible; LibraryBot/1.0)",
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        Logger.warn("Failed to fetch external PDF", {
          url: validatedUrl.href,
          status: response.status,
        });
        throw new ServiceUnavailableError(
          `Failed to fetch PDF: ${response.status}`
        );
      }

      // Check content type
      const contentType =
        response.headers.get("content-type") || "application/pdf";

      if (!contentType.includes("pdf") && !contentType.includes("octet-stream")) {
        throw new BadRequestError("URL does not point to a PDF file");
      }

      // Check content length if provided
      const contentLength = response.headers.get("content-length");
      if (contentLength && parseInt(contentLength) > MAX_PDF_SIZE) {
        throw new BadRequestError(
          `PDF too large. Maximum size is ${MAX_PDF_SIZE / (1024 * 1024)}MB`
        );
      }

      pdfBuffer = await response.arrayBuffer();

      // Validate actual size after download
      if (pdfBuffer.byteLength > MAX_PDF_SIZE) {
        throw new BadRequestError(
          `PDF too large. Maximum size is ${MAX_PDF_SIZE / (1024 * 1024)}MB`
        );
      }
    } catch (error) {
      clearTimeout(timeoutId);
      if ((error as Error).name === "AbortError") {
        throw new ServiceUnavailableError("Request timeout");
      }
      throw error;
    }

    // Return the PDF with download headers
    return new NextResponse(pdfBuffer, {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="${filename}"`,
        "Content-Length": pdfBuffer.byteLength.toString(),
        "Cache-Control": "public, max-age=3600", // Cache for 1 hour
      },
    });
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/download-pdf" });
  }
}
