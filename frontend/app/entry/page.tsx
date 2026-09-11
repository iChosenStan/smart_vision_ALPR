"use client";

import { Camera, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { StatusBadge } from "@/components/StatusBadge";
import { ApiError, api } from "@/lib/api";
import type { TicketRead } from "@/lib/types";

export default function EntryPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [ticket, setTicket] = useState<TicketRead | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleFileSelected(selected: File) {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setTicket(null);
    setError(null);
  }

  async function handleSimulateArrival() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const created = await api.registerEntry(file);
      setTicket(created);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Erro ao processar a entrada.");
      setTicket(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Entrada</h1>
        <p className="text-sm text-gray-500">
          Selecione uma imagem (câmera de entrada simulada) e clique em &quot;Simular chegada&quot;
        </p>
      </div>

      <div className="bg-surface border border-border rounded-xl p-6 space-y-4">
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFileSelected(e.target.files[0])}
        />

        <button
          onClick={() => inputRef.current?.click()}
          className="w-full border border-dashed border-border rounded-lg py-10 flex flex-col items-center gap-2 text-gray-500 hover:border-accent hover:text-accent transition-colors"
        >
          <Upload size={28} />
          <span className="text-sm">Clique para selecionar uma imagem</span>
        </button>

        {preview && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt="Pré-visualização" className="rounded-lg border border-border max-h-64 mx-auto" />
        )}

        <button
          onClick={handleSimulateArrival}
          disabled={!file || loading}
          className="w-full bg-accent hover:bg-accent/90 disabled:opacity-40 text-white rounded-lg py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors"
        >
          <Camera size={18} />
          {loading ? "Processando (IA rodando)..." : "Simular chegada"}
        </button>

        {error && <p className="text-sm text-red-400">{error}</p>}
      </div>

      {ticket && (
        <div className="bg-surface border border-border rounded-xl p-6 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-gray-500">{ticket.ticket_number}</p>
              <h2 className="text-lg font-semibold text-white">{ticket.vehicle.plate}</h2>
            </div>
            <StatusBadge status={ticket.status} />
          </div>
          <dl className="grid grid-cols-2 gap-y-1 text-sm text-gray-300">
            <dt className="text-gray-500">Marca/Modelo</dt>
            <dd>
              {ticket.vehicle.make} {ticket.vehicle.model}
            </dd>
            <dt className="text-gray-500">Cor</dt>
            <dd>{ticket.vehicle.color}</dd>
            <dt className="text-gray-500">Confiança OCR</dt>
            <dd>{(ticket.ocr_confidence * 100).toFixed(0)}%</dd>
          </dl>
          {ticket.vehicle.plate.startsWith("DESCONHECIDA-") && (
            <p className="text-xs text-amber-400">
              ⚠ A placa não foi lida com confiança — revisão manual recomendada.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
