import type {
  RequestInit,
  Response,
} from "next/dist/compiled/@edge-runtime/primitives/fetch";

type FetchInput = Parameters<typeof fetch>[0];

type FetchInit = RequestInit & {
  skipJsonParse?: boolean;
};

export class ApiError extends Error {
  public readonly status: number;
  public readonly body: unknown;

  constructor(message: string, status: number, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!text) {
    return {} as T;
  }

  try {
    return JSON.parse(text) as T;
  } catch (error) {
    throw new ApiError("Invalid JSON response", response.status, text);
  }
}

export async function fetchJson<T>(
  input: FetchInput,
  init: FetchInit = {}
): Promise<T> {
  const { skipJsonParse = false, headers, ...rest } = init;
  const baseUrl = process.env.NEXT_PUBLIC_APP_BASE_URL;
  const request =
    typeof input === "string" && baseUrl && input.startsWith("/")
      ? `${baseUrl}${input}`
      : input;

  const response = await fetch(request, {
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
    ...rest,
  });

  if (!response.ok) {
    const errorPayload = skipJsonParse
      ? undefined
      : await response
          .clone()
          .json()
          .catch(() => undefined);
    throw new ApiError(
      response.statusText || "Request failed",
      response.status,
      errorPayload
    );
  }

  if (skipJsonParse) {
    return undefined as T;
  }

  return parseJson<T>(response);
}
