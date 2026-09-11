"use client";

import { Search } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { TicketModal } from "@/components/TicketModal";
import { TicketsTable } from "@/components/TicketsTable";
import { ApiError, api } from "@/lib/api";
import type { TicketRead, TicketStatus } from "@/lib/types";

const PAGE_SIZE = 15;

export default function HistoryPage() {
  const [tickets, setTickets] = useState<TicketRead[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [plateFilter, setPlateFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState<TicketStatus | "">("");
  const [selected, setSelected] = useState<TicketRead | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      const data = await api.getHistory({
        plate: plateFilter || undefined,
        status: (statusFilter || undefined) as TicketStatus | undefined,
        page,
        page_size: PAGE_SIZE,
      });
      setTickets(data.items);
      setTotal(data.total);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Não foi possível carregar o histórico.");
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  function handleSearch(e: FormEvent) {
    e.preventDefault();
    setPage(1);
    load();
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Histórico</h1>
        <p className="text-sm text-gray-500">Todos os tickets registrados, com filtros</p>
      </div>

      <form onSubmit={handleSearch} className="flex flex-wrap gap-3">
        <input
          value={plateFilter}
          onChange={(e) => setPlateFilter(e.target.value)}
          placeholder="Filtrar por placa..."
          className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-gray-200 placeholder:text-gray-600 focus:outline-none focus:border-accent"
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as TicketStatus | "")}
          className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-accent"
        >
          <option value="">Todos os status</option>
          <option value="EM_ABERTO">Em aberto</option>
          <option value="PAGO">Pago</option>
          <option value="FINALIZADO">Finalizado</option>
        </select>
        <button
          type="submit"
          className="bg-accent hover:bg-accent/90 text-white rounded-lg px-4 py-2 text-sm font-medium flex items-center gap-2"
        >
          <Search size={16} /> Buscar
        </button>
      </form>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      <div className="bg-surface border border-border rounded-xl p-5">
        <TicketsTable tickets={tickets} onView={setSelected} />
      </div>

      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>{total} ticket(s) encontrado(s)</span>
        <div className="flex gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-3 py-1.5 rounded-lg border border-border disabled:opacity-30"
          >
            Anterior
          </button>
          <span className="px-2 py-1.5">
            {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="px-3 py-1.5 rounded-lg border border-border disabled:opacity-30"
          >
            Próxima
          </button>
        </div>
      </div>

      {selected && <TicketModal ticket={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
