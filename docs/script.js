/*
  RedHydra documentation site – improved scripts
  Extends the original behaviour with viewport‑based animations and
  minor enhancements while keeping the code light.  The theme toggle
  remains simple, switching classes on the body and allowing the CSS
  to determine which icons to display.  The carousel retains its
  auto‑switching behaviour.  Steps fade in when they enter the
  viewport using the IntersectionObserver API for better perceived
  performance and modern motion design.
*/

// Toggle between dark and light themes.  The CSS defines which
// elements appear for each state.  Additional persistence (e.g.
// localStorage) could be added if needed.
function toggleTheme() {
  document.body.classList.toggle('light');
  document.body.classList.toggle('dark');
}

// Carousel auto‑rotation: cycle through images every 4 seconds.
const slides = document.querySelectorAll('.carousel img');
let currentSlide = 0;
setInterval(() => {
  if (!slides.length) return;
  slides[currentSlide].classList.remove('active');
  currentSlide = (currentSlide + 1) % slides.length;
  slides[currentSlide].classList.add('active');
}, 4000);

// Animate steps on scroll into view using IntersectionObserver.
document.addEventListener('DOMContentLoaded', () => {
  const steps = document.querySelectorAll('.steps div');
  if (!('IntersectionObserver' in window)) {
    // Fallback for older browsers: show all steps immediately.
    steps.forEach((el) => el.classList.add('visible'));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1 }
  );
  steps.forEach((el) => observer.observe(el));
}); 
