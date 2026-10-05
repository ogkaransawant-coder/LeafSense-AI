import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    // ============================================================
    // READ FRONTEND REQUEST
    // ============================================================

    const body = await request.json();

    // ============================================================
    // FORWARD REQUEST TO FASTAPI
    // ============================================================

    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(body),

      // Do not cache AI chat responses.
      cache: "no-store",
    });

    // ============================================================
    // READ FASTAPI RESPONSE
    // ============================================================

    const contentType =
      response.headers.get("content-type") || "";

    if (!contentType.includes("application/json")) {
      return NextResponse.json(
        {
          detail: `Backend returned an unexpected response (${response.status}).`,
        },
        {
          status: response.status || 502,
        }
      );
    }

    const data = await response.json();

    // ============================================================
    // RETURN FASTAPI RESPONSE TO FRONTEND
    // ============================================================

    return NextResponse.json(data, {
      status: response.status,
    });

  } catch (error) {
    // ============================================================
    // CONNECTION ERROR
    // ============================================================

    console.error("Chat proxy error:", error);

    return NextResponse.json(
      {
        detail:
          "Unable to connect to the LeafSense AI backend.",
      },
      {
        status: 502,
      }
    );
  }
}