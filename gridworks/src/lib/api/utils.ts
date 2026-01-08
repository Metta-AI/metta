import z from "zod";

export async function fetchApi<T extends z.ZodTypeAny>(
  url: string,
  schema: T
): Promise<z.infer<T>> {
  const response = await fetch(url);
  if (response.status >= 400) {
    let detail = "Unknown error";
    if (response.headers.get("content-type") === "application/json") {
      try {
        const data = await response.json();
        if (data.detail) {
          detail = String(data.detail);
        }
      } catch (e) {
        console.error(e);
      }
    } else {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  const data = await response.json();
  return schema.parse(data);
}
