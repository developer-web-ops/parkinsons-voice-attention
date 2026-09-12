// Runtime API base for the frontend.
//
// Empty string means "same origin" — correct when the FastAPI service serves
// this page itself (the default single-service deployment on Render).
//
// If the frontend is hosted separately (e.g. on Vercel) from the Python API,
// set window.API_BASE to the backend origin, e.g.:
//   window.API_BASE = "https://your-api.onrender.com";
// This file is safe to edit at deploy time and contains no secrets.
window.API_BASE = "https://parkinsons-voice-attention.onrender.com";
