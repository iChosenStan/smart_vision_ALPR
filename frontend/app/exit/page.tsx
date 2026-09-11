"use client";

import { Camera, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { GateAnimation } from "@/components/GateAnimation";
import { ApiError, api } from "@/lib/api";
import type { ExitResponse } from "@/lib/types";

export default function ExitPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExitResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleFileSelected(selected: File) {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }

  async function handleSimulateDeparture() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const response = await api.registerExit(file);
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Erro ao processar a saída.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Saída</h1>
        <p className="text-sm text-gray-500">
          Selecione uma imagem (câmera de saída simulada) e clique em &quot;Simular saída&quot;
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
          onClick={handleSimulateDeparture}
          disabled={!file || loading}
          className="w-full bg-accent hover:bg-accent/90 disabled:opacity-40 text-white rounded-lg py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors"
        >
          <Camera size={18} />
          {loading ? "Processando (IA rodando)..." : "Simular saída"}
        </button>

        {error && <p className="text-sm text-red-400">{error}</p>}
      </div>

      <div className="bg-surface border border-border rounded-xl p-8 flex flex-col items-center justify-center">
        <GateAnimation open={result?.gate_open ?? false} denied={result !== null && !result.gate_open} />
        {result && <p className="mt-4 text-sm text-gray-400">{result.message}</p>}
      </div>
    </div>
  );
}
