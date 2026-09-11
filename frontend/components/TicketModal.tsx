"use client";

import { X } from "lucide-react";
import { useState } from "react";
import { ApiError, api, captureImageUrl } from "@/lib/api";
import type { TicketRead } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";

interface TicketModalProps {
  ticket: TicketRead;
  onClose: () => void;
  onUpdated?: (ticket: TicketRead) => void;
}

export function TicketModal({ ticket, onClose, onUpdated }: TicketModalProps) {
  const [current, setCurrent] = useState(ticket);
  const [paying, setPaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePay() {
    setPaying(true);
    setError(null);
    try {
      const updated = await api.payTicket(current.id, "dinheiro");
      setCurrent(updated);
      onUpdated?.(updated);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Erro ao processar pagamento.");
    } finally {
      setPaying(false);
    }
  }

  const vehicleImage = captureImageUrl(current.entry_image_path);
  const plateImage = captureImageUrl(current.plate_image_path);

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-panel border border-border rounded-xl w-full max-w-lg p-6 space-y-4">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-gray-500">{current.ticket_number}</p>
            <h2 className="text-lg font-semibold text-white">{current.vehicle.plate}</h2>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white">
            <X size={20} />
          </button>
        </div>

        {(vehicleImage || plateImage) && (
          <div className="grid grid-cols-2 gap-3">
            {vehicleImage && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={vehicleImage}
                alt="Veículo"
                className="rounded-lg border border-border object-cover h-32 w-full"
              />
            )}
            {plateImage && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={plateImage}
                alt="Placa"
                className="rounded-lg border border-border object-cover h-32 w-full"
              />
            )}
          </div>
        )}

        <dl className="grid grid-cols-2 gap-y-2 text-sm">
          <dt className="text-gray-500">Status</dt>
          <dd>
            <StatusBadge status={current.status} />
          </dd>

          <dt className="text-gray-500">Marca/Modelo</dt>
          <dd className="text-gray-200">
            {current.vehicle.make} {current.vehicle.model}
          </dd>

          <dt className="text-gray-500">Cor / Tipo</dt>
          <dd className="text-gray-200">
            {current.vehicle.color} / {current.vehicle.type}
          </dd>

          <dt className="text-gray-500">Confiança OCR</dt>
          <dd className="text-gray-200">{(current.ocr_confidence * 100).toFixed(0)}%</dd>

          <dt className="text-gray-500">Permanência</dt>
          <dd className="text-gray-200">{current.duration_minutes} min</dd>

          <dt className="text-gray-500">Valor</dt>
          <dd className="text-gray-200">
            {current.amount != null ? `R$ ${current.amount.toFixed(2)}` : "—"}
          </dd>
        </dl>

        {error && <p className="text-sm text-red-400">{error}</p>}

        {current.status === "EM_ABERTO" && (
          <button
            onClick={handlePay}
            disabled={paying}
            className="w-full bg-accent hover:bg-accent/90 disabled:opacity-50 text-white rounded-lg py-2.5 text-sm font-medium transition-colors"
          >
            {paying ? "Processando..." : "Pagar"}
          </button>
        )}
      </div>
    </div>
  );
}
