"use client";

interface GateAnimationProps {
  open: boolean;
  denied?: boolean;
}

export function GateAnimation({ open, denied = false }: GateAnimationProps) {
  return (
    <div className="flex flex-col items-center gap-4">
      <svg viewBox="0 0 220 140" className="w-64 h-40">
        {/* poste */}
        <rect x="18" y="20" width="14" height="110" rx="3" fill="#2a2e3f" />
        {/* base do poste */}
        <rect x="8" y="126" width="34" height="10" rx="2" fill="#1a1d29" stroke="#262a3a" />

        {/* barra da cancela — pivota em (25, 30) */}
        <g
          style={{
            transform: open ? "rotate(-78deg)" : "rotate(0deg)",
            transformOrigin: "25px 30px",
            transition: "transform 900ms cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        >
          <rect x="25" y="24" width="170" height="10" rx="4" fill={denied ? "#f87171" : "#5b8cff"} />
          <rect x="25" y="24" width="20" height="10" rx="4" fill="#facc15" />
          <rect x="165" y="24" width="20" height="10" rx="4" fill="#facc15" />
        </g>
      </svg>

      <p
        className={`text-sm font-medium ${
          denied ? "text-red-400" : open ? "text-emerald-400" : "text-gray-400"
        }`}
      >
        {denied ? "Cancela fechada — acesso negado" : open ? "Cancela aberta" : "Cancela fechada"}
      </p>
    </div>
  );
}
