"use client";

import { Camera, Leaf, Target, BarChart3 } from "lucide-react";
import type { AnalysisResult } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ImageProcessingGridProps {
  previewUrl: string | null;
  analyzing: boolean;
  result: AnalysisResult | null;
}


// ============================================================
// SEVERITY COLORS
// ============================================================

const severityColor: Record<AnalysisResult["severity"], string> = {
  None: "text-emerald-600",
  Low: "text-emerald-600",
  Moderate: "text-amber-600",
  High: "text-red-600",
};


// ============================================================
// SEVERITY BAR
// ============================================================

const severityBarStop: Record<AnalysisResult["severity"], number> = {
  None: 5,
  Low: 30,
  Moderate: 60,
  High: 90,
};


// ============================================================
// DISEASE STATUS COLOR
// ============================================================

/**
 * Disease status controls the color of the Predicted Disease card.
 *
 * Healthy   -> Green
 * Unhealthy -> Red
 * Unknown   -> Amber
 *
 * IMPORTANT:
 * Confidence does NOT determine the disease color.
 *
 * Example:
 *
 * Unhealthy + 84.5% confidence -> RED
 * Healthy   + 88.5% confidence -> GREEN
 */
function getDiseaseTone(
  prediction: string
): "emerald" | "rose" | "amber" {
  const normalizedPrediction = prediction
    .trim()
    .toLowerCase();

  if (normalizedPrediction === "healthy") {
    return "emerald";
  }

  if (normalizedPrediction === "unhealthy") {
    return "rose";
  }

  return "amber";
}


// ============================================================
// DISEASE TEXT COLOR
// ============================================================

function getDiseaseTextColor(
  prediction: string
): string {
  const normalizedPrediction = prediction
    .trim()
    .toLowerCase();

  if (normalizedPrediction === "healthy") {
    return "text-emerald-700";
  }

  if (normalizedPrediction === "unhealthy") {
    return "text-rose-700";
  }

  return "text-amber-700";
}


// ============================================================
// STAGE CARD
// ============================================================

function StageCard({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof Camera;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl bg-[var(--color-surface)]">

      <div className="flex items-center gap-2 border-b border-[var(--color-border)] px-4 py-3">
        <Icon className="h-4 w-4 text-emerald-600" />

        <span className="text-sm font-medium text-zinc-900">
          {title}
        </span>
      </div>

      <div className="min-h-[280px] flex-1 bg-zinc-100">
        {children}
      </div>

    </div>
  );
}


// ============================================================
// MAIN COMPONENT
// ============================================================

export function ImageProcessingGrid({
  previewUrl,
  analyzing,
  result,
}: ImageProcessingGridProps) {

  // ----------------------------------------------------------
  // Affected percentage
  // ----------------------------------------------------------

  const affectedPercentage = result
    ? result.affected_area.percentage
    : null;


  // ----------------------------------------------------------
  // Segmented image
  // ----------------------------------------------------------

  const segmentedSrc = result
    ? `data:image/png;base64,${result.images.segmented}`
    : null;


  // ----------------------------------------------------------
  // Affected image
  // ----------------------------------------------------------

  const affectedSrc = result
    ? `data:image/png;base64,${result.images.affected}`
    : null;


  // ----------------------------------------------------------
  // Disease status color
  //
  // IMPORTANT:
  // The color is based on Healthy / Unhealthy,
  // NOT on confidence.
  // ----------------------------------------------------------

  const diseaseTone = result
    ? getDiseaseTone(result.prediction)
    : "amber";


  const diseaseTextColor = result
    ? getDiseaseTextColor(result.prediction)
    : "text-zinc-400";


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">

      {/* =====================================================
          1. ORIGINAL IMAGE
          ===================================================== */}

      <StageCard
        icon={Camera}
        title="1. Original Image"
      >

        {previewUrl ? (

          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Original leaf"
            className="h-full w-full object-cover"
          />

        ) : (

          <Placeholder text="Upload a leaf" />

        )}

      </StageCard>


      {/* =====================================================
          2. SEGMENTED LEAF
          ===================================================== */}

      <StageCard
        icon={Leaf}
        title="2. Segmented Leaf"
      >

        {segmentedSrc ? (

          <div className="h-full w-full bg-black p-2">

            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={segmentedSrc}
              alt="Segmented leaf"
              className="h-full w-full object-contain"
            />

          </div>

        ) : analyzing ? (

          <Placeholder
            text="Segmenting…"
            dark
          />

        ) : (

          <Placeholder
            text="Awaiting image"
            dark
          />

        )}

      </StageCard>


      {/* =====================================================
          3. AFFECTED REGION
          ===================================================== */}

      <StageCard
        icon={Target}
        title="3. Affected Region"
      >

        {affectedSrc ? (

          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={affectedSrc}
            alt="Affected region"
            className="h-full w-full object-cover"
          />

        ) : analyzing ? (

          <Placeholder
            text="Analyzing…"
          />

        ) : (

          <Placeholder
            text="Awaiting image"
          />

        )}

      </StageCard>


      {/* =====================================================
          4. ANALYSIS SUMMARY
          ===================================================== */}

      <div className="flex h-full flex-col gap-3 rounded-xl bg-[var(--color-surface)] p-4">

        <div className="flex items-center gap-2 text-sm font-medium text-zinc-900">

          <BarChart3 className="h-4 w-4 text-emerald-600" />

          Analysis Summary

        </div>


        {/* ==================================================
            AFFECTED AREA
            ================================================== */}

        <SummaryCard
          tone="emerald"
          label="Affected Area"
        >

          <p className="text-2xl font-bold text-emerald-700">

            {affectedPercentage !== null
              ? `${affectedPercentage}%`
              : "—"}

          </p>

          <p className="text-xs text-emerald-700/70">
            of leaf area
          </p>

        </SummaryCard>


        {/* ==================================================
            PREDICTED DISEASE
            ==================================================
            
            COLOR IS BASED ON DISEASE STATUS:
            
            Healthy   -> GREEN
            Unhealthy -> RED
            Unknown   -> AMBER
            
            Confidence DOES NOT control this color.
            ================================================== */}

        <SummaryCard
          tone={diseaseTone}
          label="Predicted Disease"
        >

          <p
            className={cn(
              "text-base font-bold",
              diseaseTextColor
            )}
          >

            {result
              ? result.prediction
              : "—"}

          </p>


          {/* ------------------------------------------------
              Confidence is displayed separately.

              Example:

              Unhealthy
              84.5% confidence

              The 84.5% does NOT make the card green.
              ------------------------------------------------ */}

          <p
            className={cn(
              "text-xs",
              result
                ? `${diseaseTextColor}/70`
                : "text-zinc-400"
            )}
          >

            {result
              ? `${result.confidence}% confidence`
              : "Awaiting analysis"}

          </p>

        </SummaryCard>


        {/* ==================================================
            SEVERITY
            ================================================== */}

        <SummaryCard
          tone="amber"
          label="Severity Level"
        >

          <p
            className={cn(
              "text-base font-bold",
              result
                ? severityColor[result.severity]
                : "text-zinc-400"
            )}
          >

            {result
              ? result.severity
              : "—"}

          </p>


          {/* Severity scale */}

          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-gradient-to-r from-emerald-400 via-amber-400 to-red-500">

            <div
              className="h-full border-r-2 border-white/80"
              style={{
                width: `${
                  result
                    ? severityBarStop[result.severity]
                    : 0
                }%`,
              }}
            />

          </div>

        </SummaryCard>


        {/* ==================================================
            ANALYZING STATE
            ================================================== */}

        {analyzing && (

          <p className="text-center text-xs text-zinc-400">
            Running inference…
          </p>

        )}

      </div>

    </div>
  );
}


// ============================================================
// PLACEHOLDER
// ============================================================

function Placeholder({
  text,
  dark,
}: {
  text: string;
  dark?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex h-full w-full min-h-[280px] items-center justify-center text-xs",
        dark
          ? "bg-black text-white/30"
          : "bg-zinc-100 text-zinc-400"
      )}
    >
      {text}
    </div>
  );
}


// ============================================================
// SUMMARY CARD
// ============================================================

function SummaryCard({
  tone,
  label,
  children,
}: {
  tone: "emerald" | "rose" | "amber";
  label: string;
  children: React.ReactNode;
}) {

  const tones = {

    emerald:
      "bg-emerald-50 border-emerald-100",

    rose:
      "bg-rose-50 border-rose-100",

    amber:
      "bg-amber-50 border-amber-100",

  };


  return (
    <div
      className={cn(
        "rounded-lg border p-3",
        tones[tone]
      )}
    >

      <p className="text-xs font-medium text-zinc-500">
        {label}
      </p>

      <div className="mt-1">
        {children}
      </div>

    </div>
  );
}