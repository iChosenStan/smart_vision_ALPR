"use client";

import { Eye } from "lucide-react";
import type { TicketRead } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";

interface TicketsTableProps {
  tickets: TicketRead[];
  onView?: (ticket: TicketRead) => void;
}

function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" });
}

export function TicketsTable({ tickets, onView }: TicketsTableProps) {
  if (tickets.length === 0) {
    return <p className="text-sm text-gray-500 py-8 text-center">Nenhum ticket encontrado.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-500 border-b border-border">
            <th className="py-2.5 pr-4 font-medium">Ticket</th>
            <th className="py-2.5 pr-4 font-medium">Placa</th>
            <th className="py-2.5 pr-4 font-medium">Entrada</th>
            <th className="py-2.5 pr-4 font-medium">Status</th>
            <th className="py-2.5 pr-4 font-medium">Pagamento</th>
            <th className="py-2.5 pr-4 font-medium text-right">Ações</th>
          </tr>
        </thead>
        <tbody>
          {tickets.map((ticket) => (
            <tr key={ticket.id} className="border-b border-border/60 hover:bg-surface/60">
              <td className="py-2.5 pr-4 font-mono text-gray-300">{ticket.ticket_number}</td>
              <td className="py-2.5 pr-4 font-mono text-white">{ticket.vehicle.plate}</td>
              <td className="py-2.5 pr-4 text-gray-400">{formatDateTime(ticket.entry_at)}</td>
              <td className="py-2.5 pr-4">
                <StatusBadge status={ticket.status} />
              </td>
              <td className="py-2.5 pr-4 text-gray-400">
                {ticket.amount != null ? `R$ ${ticket.amount.toFixed(2)}` : "—"}
              </td>
              <td className="py-2.5 pr-4 text-right">
                <button
                  onClick={() => onView?.(ticket)}
                  className="inline-flex items-center gap-1 text-accent hover:underline text-xs"
                >
                  <Eye size={14} /> Visualizar
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
