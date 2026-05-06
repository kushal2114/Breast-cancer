/**
 * main.jsx — Entry point for the BreastAI Classifier React app.
 *
 * Mounts the App component to the DOM root element.
 */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
