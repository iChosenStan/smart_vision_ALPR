# SmartVision ALPR — Frontend

Next.js 16 (App Router) + TailwindCSS + TypeScript. Tema escuro, atualização automática do dashboard (polling a cada 5s).

## Como executar

```bash
# 1. Instalar dependências
npm install

# 2. Configurar a URL do backend (opcional — o default já aponta pro localhost:8000)
cp .env.local.example .env.local

# 3. Rodar em desenvolvimento
npm run dev
# Acesse http://localhost:3000
```

**Pré-requisito:** o backend (`backend/`) precisa estar rodando em `http://localhost:8000` (ver `docs/09_local_setup.md` e a documentação do backend) — o frontend não funciona sozinho, ele só exibe/consome os dados da API.

## Build de produção

```bash
npm run build
npm run start
```

Já validado nesta máquina de desenvolvimento: build de produção compila sem erros, TypeScript sem erros, todas as 6 rotas geradas com sucesso (`/`, `/dashboard`, `/entry`, `/exit`, `/history`).

## Estrutura

```
app/
├── layout.tsx        # layout raiz (sidebar + tema escuro)
├── page.tsx           # redireciona para /dashboard
├── dashboard/          # cards + tabela em tempo real (polling)
├── entry/               # captura de entrada, mostra o ticket criado
├── exit/                 # captura de saída, animação SVG da cancela
└── history/              # listagem paginada com filtros (placa/status)

components/
├── Sidebar.tsx, StatCard.tsx, StatusBadge.tsx
├── TicketsTable.tsx, TicketModal.tsx (com botão "Pagar")
└── GateAnimation.tsx    # cancela animada em SVG puro (fechada → abrindo → aberta)

lib/
├── api.ts     # cliente HTTP tipado (fetch) para o backend FastAPI
└── types.ts   # tipos espelhando os schemas Pydantic do backend
```
