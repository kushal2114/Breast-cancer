/**
 * UploadCard.jsx — Image upload card with drag-and-drop, modality selection,
 * preview, and analyze button.
 *
 * Supports three modes:
 *   - Mammography: single image upload
 *   - Ultrasound: single image upload
 *   - Fusion (Both): dual side-by-side upload for mammo + ultrasound
 */
import { useState, useRef, useCallback } from 'react';

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const VALID_TYPES = ['image/jpeg', 'image/png'];

function validateFile(file) {
  if (!file) return false;
  if (!VALID_TYPES.includes(file.type)) {
    alert('Please upload a JPG or PNG image.');
    return false;
  }
  if (file.size > MAX_FILE_SIZE) {
    alert('File is too large. Maximum size is 10 MB.');
    return false;
  }
  return true;
}

function readPreview(file, setter) {
  const reader = new FileReader();
  reader.onloadend = () => setter(reader.result);
  reader.readAsDataURL(file);
}

/** Reusable drop-zone component for a single image slot. */
function DropZone({ file, preview, onFile, onClear, label, accentColor, isDragOver, onDragOver, onDragLeave, onDrop, inputRef, inputId }) {
  return (
    <div className="flex-1 min-w-0">
      {!preview ? (
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => inputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-xl p-6 sm:p-10 text-center cursor-pointer transition-all duration-300 ${
            isDragOver
              ? `border-${accentColor} bg-${accentColor}/5 scale-[1.01]`
              : `border-gray-300 hover:border-${accentColor}/50 hover:bg-gray-50/50`
          }`}
          style={isDragOver ? { borderColor: `var(--color-${accentColor})`, backgroundColor: `color-mix(in srgb, var(--color-${accentColor}) 5%, transparent)` } : {}}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            onChange={(e) => onFile(e.target.files?.[0])}
            className="hidden"
            id={inputId}
          />
          <div className="flex flex-col items-center gap-2">
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors duration-300 ${
              isDragOver ? 'bg-primary/15' : 'bg-gray-100'
            }`}>
              <svg className={`w-6 h-6 transition-colors duration-300 ${isDragOver ? 'text-primary' : 'text-gray-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <p className="text-sm font-medium text-text">{label}</p>
            <p className="text-xs text-text-secondary/70">
              Drag & drop or <span className="text-primary font-medium">browse</span>
            </p>
            <p className="text-[11px] text-text-secondary/50">JPG, PNG · Max 10 MB</p>
          </div>
        </div>
      ) : (
        <div className="relative group">
          <div className="rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
            <img
              src={preview}
              alt={`${label} preview`}
              className="w-full h-48 sm:h-56 object-contain"
            />
          </div>
          <div className="mt-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5 min-w-0">
              <svg className="w-3.5 h-3.5 text-primary flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
              </svg>
              <span className="text-xs text-text-secondary truncate">{file.name}</span>
            </div>
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
              label.includes('Mammo')
                ? 'bg-primary/10 text-primary'
                : 'bg-purple-100 text-purple-700'
            }`}>
              {label}
            </span>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onClear(); }}
            className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/50 hover:bg-black/70 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 text-xs"
            title="Remove image"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}

export default function UploadCard({ onAnalyze, isLoading }) {
  // Per-stream file state
  const [mammoFile, setMammoFile] = useState(null);
  const [mammoPreview, setMammoPreview] = useState(null);
  const [usFile, setUsFile] = useState(null);
  const [usPreview, setUsPreview] = useState(null);
  const [modality, setModality] = useState('');
  const [isDragOverMammo, setIsDragOverMammo] = useState(false);
  const [isDragOverUs, setIsDragOverUs] = useState(false);
  // Single-mode drag state (reused for non-fusion)
  const [isDragOverSingle, setIsDragOverSingle] = useState(false);
  const mammoInputRef = useRef(null);
  const usInputRef = useRef(null);
  const singleInputRef = useRef(null);

  const handleMammoFile = useCallback((f) => {
    if (!validateFile(f)) return;
    setMammoFile(f);
    readPreview(f, setMammoPreview);
  }, []);

  const handleUsFile = useCallback((f) => {
    if (!validateFile(f)) return;
    setUsFile(f);
    readPreview(f, setUsPreview);
  }, []);

  // For single-mode, route to the correct stream
  const handleSingleFile = useCallback((f, mod) => {
    if (!validateFile(f)) return;
    if (mod === 'mammography') {
      setMammoFile(f);
      readPreview(f, setMammoPreview);
    } else {
      setUsFile(f);
      readPreview(f, setUsPreview);
    }
  }, []);

  const handleSubmit = () => {
    if (modality === 'fusion') {
      if (mammoFile && usFile) onAnalyze({ mammo: mammoFile, us: usFile }, 'fusion');
    } else if (modality === 'mammography') {
      if (mammoFile) onAnalyze({ mammo: mammoFile, us: null }, 'mammography');
    } else if (modality === 'ultrasound') {
      if (usFile) onAnalyze({ mammo: null, us: usFile }, 'ultrasound');
    }
  };

  const isFusion = modality === 'fusion';
  const singleFile = modality === 'mammography' ? mammoFile : usFile;
  const singlePreview = modality === 'mammography' ? mammoPreview : usPreview;

  const canSubmit = (() => {
    if (isLoading || !modality) return false;
    if (isFusion) return mammoFile && usFile;
    return singleFile != null;
  })();

  return (
    <div className="w-full max-w-3xl mx-auto animate-fade-in">
      <div className="bg-card rounded-2xl shadow-lg shadow-gray-200/50 border border-gray-100 overflow-hidden">
        <div className="p-6 sm:p-8">

          {/* ── Modality selection ─────────────────────────── */}
          <div className="mb-6">
            <label className="block text-sm font-semibold text-text mb-3">Select Imaging Mode</label>
            <div className="grid grid-cols-3 gap-3">
              {[
                { value: 'mammography', label: 'Mammography', icon: '🩻' },
                { value: 'ultrasound', label: 'Ultrasound', icon: '📡' },
                { value: 'fusion', label: 'Fusion (Both)', icon: '🔗' },
              ].map(({ value, label, icon }) => (
                <label
                  key={value}
                  className={`flex flex-col items-center justify-center gap-1.5 px-3 py-3.5 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                    modality === value
                      ? value === 'fusion'
                        ? 'border-primary bg-primary/5 shadow-sm ring-1 ring-primary/20'
                        : 'border-primary bg-primary/5 shadow-sm'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50/50'
                  }`}
                  id={`modality-${value}`}
                >
                  <input
                    type="radio"
                    name="modality"
                    value={value}
                    checked={modality === value}
                    onChange={(e) => setModality(e.target.value)}
                    className="sr-only"
                  />
                  <span className="text-lg">{icon}</span>
                  <span className={`text-xs font-medium ${modality === value ? 'text-primary' : 'text-text'}`}>
                    {label}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* ── Upload area ────────────────────────────────── */}
          {modality && (
            <>
              {isFusion ? (
                /* ── Fusion: two side-by-side dropzones ─── */
                <div className="flex gap-4">
                  <DropZone
                    file={mammoFile}
                    preview={mammoPreview}
                    onFile={handleMammoFile}
                    onClear={() => { setMammoFile(null); setMammoPreview(null); }}
                    label="Mammography"
                    accentColor="primary"
                    isDragOver={isDragOverMammo}
                    onDragOver={(e) => { e.preventDefault(); setIsDragOverMammo(true); }}
                    onDragLeave={(e) => { e.preventDefault(); setIsDragOverMammo(false); }}
                    onDrop={(e) => { e.preventDefault(); setIsDragOverMammo(false); handleMammoFile(e.dataTransfer.files?.[0]); }}
                    inputRef={mammoInputRef}
                    inputId="mammo-file-input"
                  />
                  <DropZone
                    file={usFile}
                    preview={usPreview}
                    onFile={handleUsFile}
                    onClear={() => { setUsFile(null); setUsPreview(null); }}
                    label="Ultrasound"
                    accentColor="primary"
                    isDragOver={isDragOverUs}
                    onDragOver={(e) => { e.preventDefault(); setIsDragOverUs(true); }}
                    onDragLeave={(e) => { e.preventDefault(); setIsDragOverUs(false); }}
                    onDrop={(e) => { e.preventDefault(); setIsDragOverUs(false); handleUsFile(e.dataTransfer.files?.[0]); }}
                    inputRef={usInputRef}
                    inputId="us-file-input"
                  />
                </div>
              ) : (
                /* ── Single modality dropzone ─────────── */
                <>
                  {!singlePreview ? (
                    <div
                      id="drop-zone"
                      onDrop={(e) => {
                        e.preventDefault();
                        setIsDragOverSingle(false);
                        handleSingleFile(e.dataTransfer.files?.[0], modality);
                      }}
                      onDragOver={(e) => { e.preventDefault(); setIsDragOverSingle(true); }}
                      onDragLeave={(e) => { e.preventDefault(); setIsDragOverSingle(false); }}
                      onClick={() => singleInputRef.current?.click()}
                      className={`relative border-2 border-dashed rounded-xl p-10 sm:p-14 text-center cursor-pointer transition-all duration-300 ${
                        isDragOverSingle
                          ? 'border-primary bg-primary/5 scale-[1.01]'
                          : 'border-gray-300 hover:border-primary/50 hover:bg-gray-50/50'
                      }`}
                    >
                      <input
                        ref={singleInputRef}
                        type="file"
                        accept="image/jpeg,image/png"
                        onChange={(e) => handleSingleFile(e.target.files?.[0], modality)}
                        className="hidden"
                        id="file-input"
                      />
                      <div className="flex flex-col items-center gap-3">
                        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-colors duration-300 ${
                          isDragOverSingle ? 'bg-primary/15' : 'bg-gray-100'
                        }`}>
                          <svg className={`w-8 h-8 transition-colors duration-300 ${isDragOverSingle ? 'text-primary' : 'text-gray-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                          </svg>
                        </div>
                        <div>
                          <p className="text-base font-medium text-text">
                            {isDragOverSingle ? 'Drop your image here' : 'Drag & drop your image here'}
                          </p>
                          <p className="text-sm text-text-secondary mt-1">
                            or <span className="text-primary font-medium hover:underline">click to browse</span>
                          </p>
                        </div>
                        <p className="text-xs text-text-secondary/70">
                          Supports JPG, PNG · Max 10 MB
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="relative group">
                      <div className="rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
                        <img
                          src={singlePreview}
                          alt="Uploaded preview"
                          className="w-full h-64 sm:h-72 object-contain"
                          id="image-preview"
                        />
                      </div>
                      <div className="mt-3 flex items-center justify-between">
                        <div className="flex items-center gap-2 min-w-0">
                          <svg className="w-4 h-4 text-primary flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
                          </svg>
                          <span className="text-sm text-text-secondary truncate">{singleFile.name}</span>
                        </div>
                        <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                          modality === 'mammography'
                            ? 'bg-primary/10 text-primary'
                            : 'bg-purple-100 text-purple-700'
                        }`}>
                          {modality === 'mammography' ? 'Mammography' : 'Ultrasound'}
                        </span>
                      </div>
                      <button
                        onClick={() => {
                          if (modality === 'mammography') {
                            setMammoFile(null); setMammoPreview(null);
                          } else {
                            setUsFile(null); setUsPreview(null);
                          }
                        }}
                        className="absolute top-2 right-2 w-8 h-8 rounded-full bg-black/50 hover:bg-black/70 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                        title="Remove image"
                        id="remove-image-btn"
                      >
                        ✕
                      </button>
                    </div>
                  )}
                </>
              )}
            </>
          )}

          {/* Model info */}
          <p className="text-center text-xs text-text-secondary/60 mt-5">
            {isFusion
              ? 'True Multimodal Fusion · Bidirectional Cross-Attention · DenseNet-121'
              : 'Powered by Bidirectional Cross-Attention Fusion · DenseNet-121'}
          </p>

          {/* Analyze button */}
          <button
            id="analyze-btn"
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={`w-full mt-5 py-3.5 rounded-xl font-semibold text-white text-sm transition-all duration-300 ${
              canSubmit
                ? 'bg-primary hover:bg-primary-light active:bg-primary-dark shadow-md shadow-primary/20 hover:shadow-lg hover:shadow-primary/30 hover:-translate-y-0.5'
                : 'bg-gray-300 cursor-not-allowed'
            }`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5 animate-spin-slow" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing {isFusion ? 'images' : 'image'}...
              </span>
            ) : (
              isFusion ? 'Analyze Both Images (Fusion)' : 'Analyze Image'
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
