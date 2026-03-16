import type { Metadata } from "next";
import { IBM_Plex_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-brand",
  subsets: ["latin"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-code",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Autonom Air Base",
  description: "RL-trained airbase control in a minimal turn-by-turn web UI.",
  icons: {
    icon: "/saab.png",
    shortcut: "/saab.png",
    apple: "/saab.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={[
          spaceGrotesk.variable,
          plexMono.variable,
          "min-h-screen overflow-x-hidden font-sans antialiased",
          "bg-[linear-gradient(rgba(0,0,0,0.5),rgba(0,0,0,0.5)),url('/background.webp')] bg-cover bg-center bg-fixed",
        ].join(" ")}
      >
        {children}
      </body>
    </html>
  );
}
