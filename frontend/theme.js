// theme.js — shared theme manager for BRAG AI
(function () {
  const KEY = 'bragai:theme';

  function getTheme() {
    // Default to light theme if no saved preference
    return localStorage.getItem(KEY) || 'light';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(KEY, theme);
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

  // Expose for inline onclick on multiple pages (no framework)
  window.__bragaiTheme = { getTheme, applyTheme, toggleTheme };

  // Auto-apply on load
  document.addEventListener('DOMContentLoaded', () => applyTheme(getTheme()));
})();

