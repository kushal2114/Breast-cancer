/**
 * Footer.jsx — Application footer with attribution and model info.
 *
 * Displays research demo attribution, university name, and
 * the model architecture identifier.
 */

export default function Footer() {
  return (
    <footer className="mt-12 border-t border-gray-200 bg-white/60 backdrop-blur-sm" id="app-footer">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-text-secondary/60">
          <p>Research Demo · VIT-AP University · Not for clinical use</p>
          <p>Model: Bidirectional Cross-Attention Fusion · DenseNet-121</p>
        </div>
      </div>
    </footer>
  );
}
