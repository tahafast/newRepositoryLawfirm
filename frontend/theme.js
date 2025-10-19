// theme.js — unified theme manager for BRAG AI
(function () {
  const KEY = 'bragai:theme';

  // 🎨 Light and Dark palettes using same variable names your UI uses
  const THEMES = {
    light: {
      '--bg': '#f5f7fa', // Softer off-white
      '--panel': '#ffffff',
      '--text': '#1e293b', // Proper dark gray for headings/text
      '--muted': '#475569',
      '--border': '#e2e8f0',
      '--accent': '#2563eb',
      '--card-shadow': '0 10px 30px rgba(2, 12, 27, 0.05)',
    },
    dark: {
      '--bg': '#0b1120',
      '--panel': '#111827',
      '--text': '#e2e8f0', // High-contrast light gray
      '--muted': '#94a3b8',
      '--border': '#1e293b',
      '--accent': '#3b82f6',
      '--card-shadow': '0 10px 30px rgba(0,0,0,0.4)',
    },
  };

  function getTheme() {
    return localStorage.getItem(KEY) || 'light';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(KEY, theme);

    // Apply CSS variables dynamically
    const vars = THEMES[theme];
    for (const [key, value] of Object.entries(vars)) {
      document.documentElement.style.setProperty(key, value);
    }

    // Update all toggle buttons
    const toggles = document.querySelectorAll('[data-theme-toggle]');
    toggles.forEach(btn => {
      btn.textContent = theme === 'dark' ? '🌞 Light' : '🌙 Dark';
      btn.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
    });
  }

  function toggleTheme() {
    const now = getTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(now);
  }

  window.__bragaiTheme = { getTheme, applyTheme, toggleTheme };

  document.addEventListener('DOMContentLoaded', () => {
    // Smooth fade between themes
    document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
    applyTheme(getTheme());
  });
})();
