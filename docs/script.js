// Theme toggle
function toggleTheme(){
  document.body.classList.toggle('light');
  document.body.classList.toggle('dark');
}

// Carousel
let slides = document.querySelectorAll('.carousel img');
let i = 0;
setInterval(()=>{
  slides[i].classList.remove('active');
  i = (i+1)%slides.length;
  slides[i].classList.add('active');
},4000);
