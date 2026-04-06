document.addEventListener('DOMContentLoaded', () => {
    // Cursor Follower
    const cursor = document.querySelector('.cursor-follower');
    document.addEventListener('mousemove', (e) => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
    });

    // Reveal on Scroll
    const revealElements = document.querySelectorAll('.reveal, .reveal-text, .reveal-text-delayed, .reveal-btns');
    
    const revealOnScroll = () => {
        const windowHeight = window.innerHeight;
        revealElements.forEach(el => {
            const elementTop = el.getBoundingClientRect().top;
            const elementVisible = 150;
            if (elementTop < windowHeight - elementVisible) {
                el.classList.add('active');
            }
        });
    };

    window.addEventListener('scroll', revealOnScroll);
    revealOnScroll(); // Initial check

    // Add CSS classes for text reveal effects if not already in style.css
    // Note: I'll add these to style.css in the next step if I missed them
    
    // Parallax Effect on Hero Sphere
    const sphere = document.querySelector('.glass-sphere');
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        if (sphere) {
            sphere.style.transform = `translateY(${scrolled * 0.3}px) rotate(${scrolled * 0.1}deg)`;
        }
    });

    // Mobile Menu (Simple placeholder for now)
    const navLinks = document.querySelector('.nav-links');
    const menuToggle = document.querySelector('.menu-toggle');
    
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
    }

    // Smooth Scroll for Nav Links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});
