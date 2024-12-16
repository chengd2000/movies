document.addEventListener("DOMContentLoaded", function() {
    const searchForm = document.querySelector("form");
    const queryInput = document.querySelector("input[name='query']");
    const movieList = document.querySelector(".movie-list");

    if (searchForm) {
        searchForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const query = queryInput.value;
            if (!query) return;

            try {
                const response = await fetch(`/search?query=${encodeURIComponent(query)}`);
                const movies = await response.json();
                movieList.innerHTML = "";
                movies.forEach(movie => {
                    const movieItem = document.createElement("div");
                    movieItem.classList.add("movie-item");
                    movieItem.innerHTML = `
                        <h2>${movie.title}</h2>
                        <p>Rating: ${movie.vote_average}</p>
                        <a href="/movie/${movie.id}">Details</a>
                    `;
                    movieList.appendChild(movieItem);
                });
            } catch (error) {
                console.error("Error fetching movies:", error);
            }
        });
    }
});
