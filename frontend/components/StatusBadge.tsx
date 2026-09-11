import type { TicketStatus } from "@/lib/types";

const STATUS_CONFIG: Record<TicketStatus, { label: string; className: string }> = {
  EM_ABERTO: { label: "Em aberto", className: "bg-amber-500/15 text-amber-400" },
  PAGO: { label: "Pago", className: "bg-emerald-500/15 text-emerald-400" },
  FINALIZADO: { label: "Finalizado", className: "bg-gray-500/15 text-gray-400" },
};

export function StatusBadge({ status }: { status: TicketStatus }) {
  const config = STATUS_CONFIG[status];
  return (
    <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${config.className}`}>
      {config.label}
    </span>
  );
}
