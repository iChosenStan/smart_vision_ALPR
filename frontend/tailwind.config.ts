import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        panel: "#12141c",
        surface: "#1a1d29",
        border: "#262a3a",
        accent: "#5b8cff",
      },
    },
  },
  plugins: [],
};

export default config;
