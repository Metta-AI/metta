import { NextRequest, NextResponse } from "next/server";

export async function middleware(request: NextRequest) {
  // Allow API routes and auth routes to pass through
  if (
    request.nextUrl.pathname.startsWith("/api/auth") ||
    request.nextUrl.pathname.startsWith("/api/health") ||
    request.nextUrl.pathname === "/"
  ) {
    return NextResponse.next();
  }

  // For protected routes like dashboard, redirect to home (which will handle auth)
  // We can't use auth() in middleware due to Edge Runtime limitations
  if (request.nextUrl.pathname.startsWith("/dashboard")) {
    // Check for session cookie existence (basic check)
    const sessionCookie = request.cookies.get("next-auth.session-token") ||
                         request.cookies.get("__Secure-next-auth.session-token");

    if (!sessionCookie) {
      return NextResponse.redirect(new URL("/", request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};