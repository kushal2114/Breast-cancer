/**
 * ResultsSection.jsx — Classification results and clinical information cards.
 *
 * Two-column layout (desktop) / stacked (mobile):
 *   LEFT:  Classification result with prediction label, confidence bar, probability bars
 *   RIGHT: Clinical information card with dynamic content based on prediction
 */

export default function ResultsSection({ result }) {
  if (!result) return null;

  const { prediction, confidence, probabilities, modality, inference_time_ms } = result;
  const isMalignant = prediction === 'malignant';
  const confidencePct = (confidence * 100).toFixed(1);
  const benignPct = (probabilities.benign * 100).toFixed(1);
  const malignantPct = (probabilities.malignant * 100).toFixed(1);

  return (
    <div className="w-full max-w-5xl mx-auto mt-8 animate-fade-in" id="results-section">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT — Classification Result Card */}
        <div className="bg-card rounded-2xl shadow-lg shadow-gray-200/50 border border-gray-100 overflow-hidden">
          <div className="p-6 sm:p-8">
            <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-5">
              Classification Result
            </h2>

            {/* Prediction label */}
            <div className="text-center mb-6">
              <span
                id="prediction-label"
                className={`inline-block px-6 py-2.5 rounded-xl text-xl sm:text-2xl font-bold tracking-wide ${
                  isMalignant
                    ? 'bg-coral/10 text-coral border-2 border-coral/20'
                    : 'bg-green/10 text-green border-2 border-green/20'
                }`}
              >
                {prediction.toUpperCase()}
              </span>
            </div>

            {/* Confidence */}
            <div className="mb-6">
              <div className="flex justify-between items-baseline mb-2">
                <span className="text-sm font-medium text-text">Confidence</span>
                <span className={`text-lg font-bold ${isMalignant ? 'text-coral' : 'text-green'}`}>
                  {confidencePct}%
                </span>
              </div>
              <div className="w-full h-3 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${
                    isMalignant ? 'bg-coral' : 'bg-green'
                  }`}
                  style={{ width: `${confidencePct}%` }}
                />
              </div>
            </div>

            {/* Probability bars */}
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-baseline mb-1.5">
                  <span className="text-sm text-text-secondary flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-green inline-block" />
                    Benign
                  </span>
                  <span className="text-sm font-semibold text-text">{benignPct}%</span>
                </div>
                <div className="w-full h-2.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green rounded-full transition-all duration-700 ease-out"
                    style={{ width: `${benignPct}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between items-baseline mb-1.5">
                  <span className="text-sm text-text-secondary flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-coral inline-block" />
                    Malignant
                  </span>
                  <span className="text-sm font-semibold text-text">{malignantPct}%</span>
                </div>
                <div className="w-full h-2.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-coral rounded-full transition-all duration-700 ease-out"
                    style={{ width: `${malignantPct}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Meta info */}
            <div className="mt-6 pt-4 border-t border-gray-100 flex justify-between text-xs text-text-secondary/60">
              <span>
                {modality === 'fusion'
                  ? 'Multimodal fusion analysis (mammo + ultrasound)'
                  : `Single-modality analysis (${modality})`}
              </span>
              <span>{inference_time_ms.toFixed(0)} ms</span>
            </div>
          </div>
        </div>

        {/* RIGHT — Clinical Information Card */}
        <div className="bg-card rounded-2xl shadow-lg shadow-gray-200/50 border border-gray-100 overflow-hidden">
          <div className="p-6 sm:p-8">
            <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-5">
              What This Means
            </h2>

            {isMalignant ? (
              <>
                <div className="flex items-start gap-3 mb-4">
                  <span className="text-2xl flex-shrink-0 mt-0.5">🔴</span>
                  <div>
                    <h3 className="font-display text-xl font-bold text-coral">
                      Suspicious Finding — Seek Medical Attention
                    </h3>
                  </div>
                </div>
                <p className="text-sm text-text-secondary leading-relaxed mb-5">
                  A malignant result indicates the model detected features associated with cancerous tissue. This is <strong className="text-text">NOT a confirmed diagnosis</strong>. Malignant predictions must be validated through professional radiological review and, if confirmed, biopsy. Please do not delay in consulting a qualified oncologist or radiologist.
                </p>
                <div className="space-y-3 mb-6">
                  <h4 className="text-sm font-semibold text-text">Recommended Next Steps</h4>
                  <div className="space-y-2.5">
                    <div className="flex items-start gap-2.5 text-sm text-text-secondary">
                      <span className="flex-shrink-0">🏥</span>
                      <span>Consult a radiologist or oncologist promptly</span>
                    </div>
                    <div className="flex items-start gap-2.5 text-sm text-text-secondary">
                      <span className="flex-shrink-0">📋</span>
                      <span>Bring this result and your imaging to your appointment</span>
                    </div>
                    <div className="flex items-start gap-2.5 text-sm text-text-secondary">
                      <span className="flex-shrink-0">🔬</span>
                      <span>Ask your doctor about biopsy and further diagnostics</span>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-start gap-3 mb-4">
                  <span className="text-2xl flex-shrink-0 mt-0.5">🟢</span>
                  <div>
                    <h3 className="font-display text-xl font-bold text-green">
                      Likely Non-Cancerous Finding
                    </h3>
                  </div>
                </div>
                <p className="text-sm text-text-secondary leading-relaxed mb-5">
                  A benign result suggests the identified lesion does not display malignant characteristics. Common benign conditions include fibroadenomas, cysts, and other non-threatening tissue changes. While reassuring, benign findings should still be discussed with a healthcare provider, as some may require monitoring or follow-up imaging over time.
                </p>
                <div className="space-y-3 mb-6">
                  <h4 className="text-sm font-semibold text-text">Recommended Next Steps</h4>
                  <div className="space-y-2.5">
                    <div className="flex items-start gap-2.5 text-sm text-text-secondary">
                      <span className="flex-shrink-0">✅</span>
                      <span>Discuss findings with your doctor</span>
                    </div>
                    <div className="flex items-start gap-2.5 text-sm text-text-secondary">
                      <span className="flex-shrink-0">📅</span>
                      <span>Schedule routine follow-up as advised</span>
                    </div>
                    <div className="flex items-start gap-2.5 text-sm text-text-secondary">
                      <span className="flex-shrink-0">🔍</span>
                      <span>Continue regular screening as recommended</span>
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Warning footer */}
            <div className="bg-amber-warning/60 rounded-xl p-3.5 border border-amber-300/40">
              <p className="text-xs text-amber-text leading-relaxed">
                ⚠️ AI predictions are not a substitute for clinical diagnosis. This tool is a research prototype.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
