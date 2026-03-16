import type { Config } from "tailwindcss";
import defaultTheme from "tailwindcss/defaultTheme";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "transparent",
        foreground: "#ecf7ff",
        card: "#08121fd9",
        border: "#54c1ff33",
        muted: "#22384c",
        "muted-foreground": "#9cb5cb",
        primary: "#42f5c5",
        "primary-foreground": "#041018",
        destructive: "#ff637d",
        "status-available": "#42f5c5",
        "status-mission": "#67a6ff",
        "status-transit": "#ffd166",
        "status-maintenance": "#ff9966",
      },
      fontFamily: {
        sans: ["var(--font-brand)", ...defaultTheme.fontFamily.sans],
        mono: ["var(--font-code)", ...defaultTheme.fontFamily.mono],
      },
      boxShadow: {
        panel: "0 24px 60px rgba(0, 0, 0, 0.45)",
      },
    },
  },
};

export default config;
