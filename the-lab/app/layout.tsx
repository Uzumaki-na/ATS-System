import type { Metadata } from "next";
import { Crimson_Pro, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { NavBar } from "@/components/navbar";

const crimsonPro = Crimson_Pro({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-crimson-pro",
  display: "swap",
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-ibm-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "The Lab — TriadRank ATS Showcase",
  description:
    "A transparent, three-tier ML pipeline for resume scoring. See exactly how your resume is scored — no black box.",
  openGraph: {
    title: "The Lab — TriadRank ATS Showcase",
    description:
      "A transparent, three-tier ML pipeline for resume scoring. No black box.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      data-scroll-behavior="smooth"
      className={`${crimsonPro.variable} ${ibmPlexMono.variable} h-full`}
    >
      <body className="min-h-dvh flex flex-col antialiased">
        <NavBar />
        <main className="animate-fade-in-up flex-1 flex flex-col">{children}</main>
      </body>
    </html>
  );
}
