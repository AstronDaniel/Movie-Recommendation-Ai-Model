document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.selectable-card');
    const fabBar = document.getElementById('fab-bar');
    const countSpan = document.getElementById('selected-count');
    const recommendBtn = document.getElementById('btn-recommend-interactive');
    
    let selectedMovies = new Set();

    cards.forEach(card => {
        card.addEventListener('click', () => {
            const movieId = card.dataset.id;
            const title = card.dataset.title;
            
            if (selectedMovies.has(movieId)) {
                selectedMovies.delete(movieId);
                card.classList.remove('selected');
            } else {
                if (selectedMovies.size >= 5) {
                    alert('You can only select up to 5 movies!');
                    return;
                }
                selectedMovies.add(movieId);
                card.classList.add('selected');
            }
            
            updateUI();
        });
    });

    function updateUI() {
        const count = selectedMovies.size;
        countSpan.textContent = count;
        
        if (count > 0) {
            fabBar.classList.add('visible');
        } else {
            fabBar.classList.remove('visible');
        }
    }

    if (recommendBtn) {
        recommendBtn.addEventListener('click', async () => {
            if (selectedMovies.size === 0) return;
            
            recommendBtn.textContent = 'Thinking...';
            recommendBtn.disabled = true;
            
            try {
                const response = await fetch('/recommend_interactive', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        movie_ids: Array.from(selectedMovies)
                    })
                });
                
                if (response.ok) {
                    const html = await response.text();
                    document.open();
                    document.write(html);
                    document.close();
                } else {
                    alert('Something went wrong. Please try again.');
                    recommendBtn.textContent = 'Get Recommendations';
                    recommendBtn.disabled = false;
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Network error. Please try again.');
                recommendBtn.textContent = 'Get Recommendations';
                recommendBtn.disabled = false;
            }
        });
    }
});
