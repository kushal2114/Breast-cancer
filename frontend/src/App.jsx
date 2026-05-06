/**
 * App.jsx — Main application shell for the BreastAI Classifier.
 *
 * Orchestrates the full user flow:
 *   Upload → Loading → Results/Error → Reset
 *
 * Supports single-modality (mammography or ultrasound) and
 * dual-modality fusion (both) analysis modes.
 *
 * State management:
 *   - appState: 'upload' | 'loading' | 'results' | 'error'
 *   - result: prediction response from the API
 *   - errorMessage: user-friendly error string
 */
import { useState, useCallback } from 'react';
import Header from './components/Header';
import UploadCard from './components/UploadCard';
import ResultsSection from './components/ResultsSection';
import ExplainabilitySection from './components/ExplainabilitySection';
import Footer from './components/Footer';
import { analyzeImage, analyzeFusionImages } from './api';

export default function App() {
  const [appState, setAppState] = useState('upload'); // upload | loading | results | error
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const handleAnalyze = useCallback(async (files, modality) => {
    setAppState('loading');
    setResult(null);
    setErrorMessage('');

    try {
      let data;
      if (modality === 'fusion') {
        data = await analyzeFusionImages(files.mammo, files.us);
      } else {
        const file = modality === 'mammography' ? files.mammo : files.us;
        data = await analyzeImage(file, modality);
      }
      setResult(data);
      setAppState('results');
    } catch (err) {
      setErrorMessage(
        err.message || 'Analysis failed. Please check your connection and try again.'
      );
      setAppState('error');
    }
  }, []);

  const handleReset = useCallback(() => {
    setAppState('upload');
    setResult(null);
    setErrorMessage('');
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-bg">
      <Header />

      <main className="flex-1 px-4 sm:px-6 py-8 sm:py-12">
        {/* Upload / Loading state */}
        {(appState === 'upload' || appState === 'loading') && (
          <div className="relative">
            <UploadCard
              onAnalyze={handleAnalyze}
              isLoading={appState === 'loading'}
            />

            {/* Loading overlay */}
            {appState === 'loading' && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/70 backdrop-blur-sm rounded-2xl z-10" id="loading-overlay">
                <div className="flex flex-col items-center gap-4 p-8">
                  <div className="w-12 h-12 rounded-full border-3 border-gray-200 border-t-primary animate-spin-slow" />
                  <div className="text-center">
                    <p className="text-sm font-semibold text-text">Analyzing image...</p>
                    <p className="text-xs text-text-secondary mt-1">This may take a few seconds</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results state */}
        {appState === 'results' && result && (
          <>
            <ResultsSection result={result} />
            <ExplainabilitySection result={result} />

            {/* Reset button */}
            <div className="flex justify-center mt-8">
              <button
                id="reset-btn"
                onClick={handleReset}
                className="px-8 py-3 rounded-xl font-semibold text-sm text-primary bg-primary/10 border-2 border-primary/20 hover:bg-primary/15 hover:border-primary/30 transition-all duration-200 hover:-translate-y-0.5"
              >
                Analyze Another Image
              </button>
            </div>
          </>
        )}

        {/* Error state */}
        {appState === 'error' && (
          <div className="w-full max-w-2xl mx-auto animate-fade-in" id="error-card">
            <div className="bg-error-bg rounded-2xl border-2 border-error-border/30 p-6 sm:p-8 text-center">
              <div className="w-14 h-14 rounded-full bg-error-border/10 flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-error-border" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
              </div>
              <h3 className="text-lg font-bold text-error-text mb-2">Analysis Failed</h3>
              <p className="text-sm text-error-text/80 mb-6">{errorMessage}</p>
              <button
                onClick={handleReset}
                className="px-6 py-2.5 rounded-xl font-semibold text-sm text-white bg-error-border hover:bg-red-600 transition-colors duration-200"
                id="error-retry-btn"
              >
                Try Again
              </button>
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
