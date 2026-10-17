import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        warm: {
          bg: "#FAF8F5",
          surface: "#FFFFFF",
          soft: "#F5F0EA",
          text: "#28231F",
          muted: "#766B62",
          border: "#E8E1D9",
          accent: "#D97745",
          "accent-hover": "#B95F34",
          highlight: "#FBEBDD",
          success: "#2D6A4F",
          warning: "#B45309",
          error: "#C53030",
        },
      },
    },
  },
  plugins: [],
};
export default config;
