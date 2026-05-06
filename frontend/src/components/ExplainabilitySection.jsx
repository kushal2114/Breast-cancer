/**
 * ExplainabilitySection.jsx — Collapsible Grad-CAM++ heatmap viewer.
 *
 * Displays a single Grad-CAM++ heatmap for single-modality predictions,
 * or side-by-side heatmaps (Mammography + Ultrasound) for fusion mode.
 */
import { useState } from 'react';

export default function ExplainabilitySection({ result }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!result) return null;

  const { gradcam_image, gradcam_mammo, gradcam_us, modality } = result;
  const isFusion = modality === 'fusion' && gradcam_mammo && gradcam_us;
  const hasSingle = !!gradcam_image;

  if (!isFusion && !hasSingle) return null;

  return (
    <div className="w-full max-w-5xl mx-auto mt-6 animate-fade-in" id="explainability-section">
      <div className="bg-card rounded-2xl shadow-lg shadow-gray-200/50 border border-gray-100 overflow-hidden">
        {/* Toggle header */}
        <button
          id="explainability-toggle"
          onClick={() => setIsOpen((prev) => !prev)}
          className="w-full px-6 sm:px-8 py-4 flex items-center justify-between hover:bg-gray-50/50 transition-colors duration-200"
        >
          <div className="flex items-center gap-2.5">
            <span className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <svg className="w-4 h-4 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </span>
            <span className="text-sm font-semibold text-text">
              {isFusion ? 'Show Model Explanations (Both Streams)' : 'Show Model Explanation'}
            </span>
          </div>
          <svg
            className={`w-5 h-5 text-text-secondary transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
          </svg>
        </button>

        {/* Collapsible content */}
        <div
          className={`transition-all duration-400 ease-in-out overflow-hidden ${
            isOpen ? 'max-h-[1200px] opacity-100' : 'max-h-0 opacity-0'
          }`}
        >
          <div className="px-6 sm:px-8 pb-6 sm:pb-8">
            <div className="border-t border-gray-100 pt-5">

              {isFusion ? (
                /* ── Fusion: side-by-side heatmaps ───────── */
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2 text-center">
                      🩻 Mammography Stream
                    </h4>
                    <div className="rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
                      <img
                        src={`data:image/png;base64,${gradcam_mammo}`}
                        alt="Grad-CAM++ heatmap for mammography stream"
                        className="w-full object-contain max-h-80"
                        id="gradcam-mammo"
                      />
                    </div>
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2 text-center">
                      📡 Ultrasound Stream
                    </h4>
                    <div className="rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
                      <img
                        src={`data:image/png;base64,${gradcam_us}`}
                        alt="Grad-CAM++ heatmap for ultrasound stream"
                        className="w-full object-contain max-h-80"
                        id="gradcam-us"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                /* ── Single modality heatmap ──────────────── */
                <div className="rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
                  <img
                    src={`data:image/png;base64,${gradcam_image}`}
                    alt="Grad-CAM++ heatmap showing regions the model focused on"
                    className="w-full object-contain max-h-96"
                    id="gradcam-image"
                  />
                </div>
              )}

              <p className="text-xs text-text-secondary/70 mt-3 text-center leading-relaxed">
                {isFusion
                  ? 'Highlighted regions show where the model focused in each imaging stream. Warmer colors (red/yellow) indicate higher attention. Generated using Grad-CAM++ on DenseNet-121.'
                  : 'Highlighted regions show where the model focused when making its prediction. Warmer colors (red/yellow) indicate higher attention. Generated using Grad-CAM++ on DenseNet-121.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
