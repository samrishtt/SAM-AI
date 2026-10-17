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
        sam: {
          50: "#f0f4ff",
          100: "#dbe5ff",
          200: "#bfd0ff",
          300: "#93b0ff",
          400: "#6085ff",
          500: "#3b5eff",
          600: "#243ff5",
          700: "#1a2edb",
          800: "#1826b0",
          900: "#19258b",
          950: "#101655",
        },
      },
    },
  },
  plugins: [],
};
export default config;
