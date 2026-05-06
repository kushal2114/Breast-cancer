/**
 * Header.jsx — App header with branding, health indicator, and disclaimer.
 *
 * Displays the app name, tagline, "Research Demo" pill badge,
 * a live backend health dot (green/red), and an amber disclaimer strip.
 */
import { useEffect, useState } from 'react';
import { checkHealth } from '../api';

export default function Header() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let mounted = true;

    async function pollHealth() {
      try {
        const data = await checkHealth();
        if (mounted) setIsHealthy(data.status === 'ok');
      } catch {
        if (mounted) setIsHealthy(false);
      } finally {
        if (mounted) setChecking(false);
      }
    }

    pollHealth();
    const interval = setInterval(pollHealth, 15000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  return (
    <header>
      {/* Main header bar */}
      <div className="bg-white/80 backdrop-blur-md border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          {/* Left: Branding */}
          <div>
            <h1 className="font-display text-2xl sm:text-3xl font-bold text-primary tracking-tight">
              BreastAI Classifier
            </h1>
            <p className="text-sm text-text-secondary mt-0.5">
              Multimodal Breast Cancer Classification
            </p>
          </div>

          {/* Right: Badge + Health dot */}
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20">
              Research Demo
            </span>
            <div className="flex items-center gap-1.5" title={isHealthy ? 'Backend connected' : 'Backend unreachable'}>
              <span
                id="health-dot"
                className={`w-2.5 h-2.5 rounded-full transition-colors duration-300 ${
                  checking
                    ? 'bg-gray-400'
                    : isHealthy
                      ? 'bg-green-500 animate-pulse-dot'
                      : 'bg-red-500'
                }`}
              />
              <span className="text-xs text-text-secondary hidden sm:inline">
                {checking ? 'Checking...' : isHealthy ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer strip */}
      <div className="bg-amber-warning border-b border-amber-300">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-2.5">
          <p className="text-xs sm:text-sm text-amber-text text-center leading-snug">
            <span className="font-semibold">⚠️ Disclaimer:</span> This tool is for research purposes only and does not constitute medical advice. Consult a qualified healthcare professional for diagnosis.
          </p>
        </div>
      </div>
    </header>
  );
}
