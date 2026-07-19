"use client";

interface ScoreRadarProps {
  semantic: number;
  syntax: number;
  experience: number;
  education: number;
  keywords: number;
}

const CX = 140;
const CY = 140;
const R = 100;
const LABEL_R = 128;
const GRID = [0.25, 0.5, 0.75, 1.0];
const LABELS = ["Semantic", "Syntax", "Experience", "Education", "Keywords"];

function vertex(i: number, r: number): [number, number] {
  const a = (2 * Math.PI * i) / 5 - Math.PI / 2;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}

function pentPath(r: number): string {
  let d = "";
  for (let i = 0; i < 5; i++) {
    const [x, y] = vertex(i, r);
    d += `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }
  return d + " Z";
}

function dataPath(scores: number[]): string {
  let d = "";
  for (let i = 0; i < 5; i++) {
    const [x, y] = vertex(i, scores[i] * R);
    d += `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }
  return d + " Z";
}

export default function ScoreRadar({
  semantic,
  syntax,
  experience,
  education,
  keywords,
}: ScoreRadarProps) {
  const scores = [semantic, syntax, experience, education, keywords];

  return (
    <svg
      viewBox="0 0 280 280"
      width={280}
      height={280}
      xmlns="http://www.w3.org/2000/svg"
      style={{ fontFamily: "var(--font-mono), IBM Plex Mono, monospace" }}
    >
      {/* Grid pentagons */}
      {GRID.map((g) => (
        <path
          key={g}
          d={pentPath(g * R)}
          fill="none"
          stroke={g === 1 ? "#0000FF" : "#222"}
          strokeWidth={g === 1 ? 1.5 : 1}
          strokeDasharray={g === 1 ? 1000 : undefined}
          style={
            g === 1
              ? { animation: "radar-draw 1s ease-out forwards" }
              : undefined
          }
        />
      ))}

      {/* Axis lines */}
      {Array.from({ length: 5 }, (_, i) => {
        const [x, y] = vertex(i, R);
        return (
          <line
            key={`axis-${i}`}
            x1={CX}
            y1={CY}
            x2={x}
            y2={y}
            stroke="#333"
            strokeWidth={1}
            strokeDasharray={1000}
            style={{
              animation: "radar-draw 1s ease-out forwards",
              animationDelay: "0.2s",
              opacity: 0.4,
            }}
          />
        );
      })}

      {/* Data polygon */}
      <path
        d={dataPath(scores)}
        fill="#0000FF"
        fillOpacity={0.15}
        stroke="#0000FF"
        strokeWidth={2}
        style={{ animation: "fade-in 0.6s ease-out 0.4s both" }}
      />

      {/* Data points */}
      {scores.map((s, i) => {
        const [x, y] = vertex(i, s * R);
        return (
          <circle
            key={`dot-${i}`}
            cx={x}
            cy={y}
            r={3.5}
            fill="#0000FF"
            style={{ animation: "fade-in 0.4s ease-out 0.6s both" }}
          />
        );
      })}

      {/* Labels */}
      {LABELS.map((label, i) => {
        const [x, y] = vertex(i, LABEL_R);
        const dx = x - CX;
        const textAnchor =
          Math.abs(dx) < 5 ? "middle" : dx > 0 ? "start" : "end";
        const dy = y - CY;
        const dominantBaseline =
          Math.abs(dy) < 5 ? "middle" : dy < 0 ? "auto" : "hanging";
        return (
          <text
            key={`label-${i}`}
            x={x}
            y={y}
            textAnchor={textAnchor}
            dominantBaseline={dominantBaseline}
            fill="#666"
            fontSize={11}
            style={{ animation: "fade-in 0.4s ease-out 0.5s both" }}
          >
            {label}
          </text>
        );
      })}
    </svg>
  );
}
