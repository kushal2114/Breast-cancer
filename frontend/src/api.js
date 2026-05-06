/**
 * api.js — API client for the BreastAI Classifier backend.
 *
 * Provides functions to check backend health and submit images
 * for classification. All calls have a 60-second timeout to
 * accommodate CPU-based inference.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TIMEOUT_MS = 60000; // 60 seconds

/**
 * Check the backend health status.
 * @returns {Promise<Object>} Health response with status, model, backbone, device.
 */
export async function checkHealth() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${API_URL}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Health check timed out. Is the backend running?');
    }
    throw error;
  }
}

/**
 * Submit an image for classification.
 * @param {File} imageFile - The image file to classify.
 * @param {string} modality - "mammography" or "ultrasound".
 * @returns {Promise<Object>} Prediction response with results and gradcam.
 */
export async function analyzeImage(imageFile, modality) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('modality', modality);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Analysis failed with status ${response.status}`
      );
    }
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(
        'Analysis timed out. This may happen on CPU. Please try again.'
      );
    }
    throw error;
  }
}

/**
 * Submit both mammography and ultrasound images for fusion classification.
 * @param {File} mammoFile - The mammography image file.
 * @param {File} usFile - The ultrasound image file.
 * @returns {Promise<Object>} Prediction response with results and dual gradcam.
 */
export async function analyzeFusionImages(mammoFile, usFile) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  const formData = new FormData();
  formData.append('mammo_image', mammoFile);
  formData.append('us_image', usFile);

  try {
    const response = await fetch(`${API_URL}/predict-fusion`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Fusion analysis failed with status ${response.status}`
      );
    }
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(
        'Analysis timed out. This may happen on CPU. Please try again.'
      );
    }
    throw error;
  }
}
