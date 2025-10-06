import { auth } from "@/lib/auth";
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const { sessionToken } = await request.json();

    if (!sessionToken) {
      return NextResponse.json(
        { error: "Session token is required" },
        { status: 400 }
      );
    }

    // For NextAuth.js, we can validate by checking the current session
    // This is a simplified approach - in production you might want more sophisticated validation
    const session = await auth();

    if (!session?.user) {
      return NextResponse.json(
        { valid: false, error: "Invalid session" },
        { status: 401 }
      );
    }

    return NextResponse.json({
      valid: true,
      user: {
        id: session.user.id,
        email: session.user.email,
        name: session.user.name,
      },
    });
  } catch (error) {
    console.error("Session validation error:", error);
    return NextResponse.json(
      { valid: false, error: "Validation failed" },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const session = await auth();

    if (!session?.user) {
      return NextResponse.json(
        { valid: false, error: "No active session" },
        { status: 401 }
      );
    }

    return NextResponse.json({
      valid: true,
      user: {
        id: session.user.id,
        email: session.user.email,
        name: session.user.name,
      },
    });
  } catch (error) {
    console.error("Session validation error:", error);
    return NextResponse.json(
      { valid: false, error: "Validation failed" },
      { status: 500 }
    );
  }
}