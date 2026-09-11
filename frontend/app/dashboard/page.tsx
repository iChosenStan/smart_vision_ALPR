"use client";

import { useCallback, useEffect, useState } from "react";
import { Car, CircleDollarSign, Clock, ParkingSquare } from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { TicketModal } from "@/components/TicketModal";
import { TicketsTable } from "@/components/TicketsTable";
import { ApiError, api } from "@/lib/api";
import type { DashboardResponse, TicketRead } from "@/lib/types";

const POLL_INTERVAL_MS = 5000;

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardResponse | null>(null);
  const [tickets, setTickets] = useState<TicketRead[]>([]);
  const [selected, setSelected] = useState<TicketRead | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [dashboardData, ticketsData] = await Promise.all([
        api.getDashboard(),
        api.listTickets({ page: 1, page_size: 10 }),
      ]);
      setSummary(dashboardData);
      setTickets(ticketsData.items);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Não foi possível conectar à API.");
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [refresh]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Dashboard</h1>
        <p className="text-sm text-gray-500">Visão geral do estacionamento em tempo real</p>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        <StatCard label="Veículos cadastrados" value={summary?.total_vehicles ?? "—"} icon={Car} accent="blue" />
        <StatCard
          label="No estacionamento"
          value={summary?.vehicles_parked ?? "—"}
          icon={ParkingSquare}
          accent="yellow"
        />
        <StatCard label="Tickets pagos" value={summary?.tickets_paid ?? "—"} icon={CircleDollarSign} accent="green" />
        <StatCard label="Tickets pendentes" value={summary?.tickets_pending ?? "—"} icon={Clock} accent="red" />
        <StatCard
          label="Receita simulada"
          value={summary ? `R$ ${summary.simulated_revenue.toFixed(2)}` : "—"}
          icon={CircleDollarSign}
          accent="green"
        />
      </div>

      {summary && (
        <p className="text-xs text-gray-500">
          {summary.available_spots} de {summary.total_spots} vagas disponíveis
        </p>
      )}

      <div className="bg-surface border border-border rounded-xl p-5">
        <h2 className="text-sm font-medium text-gray-300 mb-4">Tickets recentes</h2>
        <TicketsTable tickets={tickets} onView={setSelected} />
      </div>

      {selected && (
        <TicketModal
          ticket={selected}
          onClose={() => setSelected(null)}
          onUpdated={(updated) => {
            setSelected(updated);
            refresh();
          }}
        />
      )}
    </div>
  );
}
