import type { Metadata } from "next";
import { Fraunces, IBM_Plex_Mono, Source_Sans_3 } from "next/font/google";
import "./globals.css";

const sans = Source_Sans_3({
  variable: "--font-sans-text",
  subsets: ["latin"],
});

const serif = Fraunces({
  variable: "--font-serif-text",
  subsets: ["latin"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-mono-text",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Readhead",
  description: "Compile a readout head from the shape of a question and run it on a frozen open model.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${serif.variable} ${mono.variable} font-sans antialiased`}>{children}</body>
    </html>
  );
}
