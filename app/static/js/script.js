const searchForm = document.querySelector("form");
const queryInput = document.querySelector("input[name='query']");
const movieList = document.querySelector(".movie-list");
const resultsDiv = document.querySelector("#results");
const checkboxes = document.querySelectorAll('.form-check-input'); 
const from_date_input=document.getElementById("from_date");
const until_date_input=document.getElementById("until_date");
const rangeInput = document.getElementById('rangeInput'); 
const search=document.getElementById("search");
const actor1Input=document.getElementById("actor1");
const actor2Input=document.getElementById("actor2");
async function searching() {
            const query = queryInput.value;
            const from_date=from_date_input.value;
            const until_date=until_date_input.value;
            const min_rating=rangeInput.value;
            const actor1=actor1Input.value;
            const actor2=actor2Input.value;
            const checkedValues = []; 
            checkboxes.forEach(checkbox => { 
                if (checkbox.checked) { 
                    console.log(checkbox.value);
                    checkedValues.push(checkbox.value); 
                }
             });
    if (query)
        window.open(`/search?query=${encodeURIComponent(query)}&genres=${encodeURIComponent(checkedValues)}&from_date=${encodeURIComponent(from_date)}&until_date=${encodeURIComponent(until_date)}&min_rating=${encodeURIComponent(min_rating)}&actor1=${encodeURIComponent(actor1)}&actor2=${encodeURIComponent(actor2)}`,"_self")
    else
        window.open(`/search?genres=${encodeURIComponent(checkedValues)}&from_date=${encodeURIComponent(from_date)}&until_date=${encodeURIComponent(until_date)}&min_rating=${encodeURIComponent(min_rating)}&actor1=${encodeURIComponent(actor1)}&actor2=${encodeURIComponent(actor2)}`,"_self")
}
search.addEventListener('click', searching);
document.getElementById("loadDataButton").addEventListener('click',() => {
    window.open("/graphs",_self);
});
document.getElementById("loadMLButton").addEventListener('click',() => {
    window.open("/MLalgo",_self);
});
// document.addEventListener("DOMContentLoaded", function() {
//     // משתנים לשני החלקים
    

//     // פונקציה עבור חיפוש סרטים
//     // if (searchForm) {
//     //     searchForm.addEventListener("submit", async (event) => {
//     //         event.preventDefault();
            
             
//     //         if (!query) return;

//     //         try {
//     //             const response = await fetch(`/search?query=${encodeURIComponent(query)}&genres=${encodeURIComponent(checkedValues)}&from_date=${encodeURIComponent(from_date)}&until_date=${encodeURIComponent(until_date)}&min_rating=${encodeURIComponent(min_rating)}`);
//     //             const movies = await response.json();
//     //             movieList.innerHTML = "";
//     //             movies.forEach(movie => {
//     //                 const movieItem = document.createElement("div");
//     //                 movieItem.classList.add("movie-item");
//     //                 movieItem.innerHTML = `
//     //                     <h2>${movie.title}</h2>
//     //                     <p>Rating: ${movie.vote_average}</p>
//     //                     <a href="/movie/${movie.id}">Details</a>
//     //                 `;
//     //                 movieList.appendChild(movieItem);
//     //             });
//     //         } catch (error) {
                
//     //             console.error("Error fetching movies:", error);
//     //         }
//     //     });
//     }

//     // פונקציה עבור טעינת נתונים לאלגוריתם
//     if (loadDataButton) {
//         loadDataButton.addEventListener("click", async () => {
//             try {
//                 const response = await fetch("/fetch_data_for_algorithm");
//                 const result = await response.json();
//                 resultsDiv.innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
//             } catch (error) {
//                 console.error("Error fetching data:", error);
//             }
//         });
//     }

//  });



// script.js

// Get elements
const openPopupBtn = document.getElementById('openPopupBtn');
const closePopupBtn = document.getElementById('closePopupBtn');
const popup = document.getElementById('popup');
//const overlay = document.getElementById('overlay_popup');

// Function to open the popup
function openPopup() {
    popup.style.display = 'block';
    //overlay.style.display = 'block';
}

// Function to close the popup
function closePopup() {
    popup.style.display = 'none';
    //overlay.style.display = 'none';
}

// Event listeners
openPopupBtn.addEventListener('click', openPopup);
closePopupBtn.addEventListener('click', closePopup);
//overlay.addEventListener('click', closePopup);
const clearValues=document.getElementById('clearValues');

function clearVals() {
    actor1Input.value=""
    actor2Input.value=""
    from_date_input.value="";
    until_date_input.value="";
    rangeInput.value=""; 
    checkboxes.forEach(checkbox => { 
        checkbox.checked=false;
     });
}
clearValues.addEventListener('click', clearVals);




function createAutoScroll(elementId, scrollStep) {
    const scrollingDiv = document.getElementById(elementId);
    let scrollAmount = 0;
    function autoScroll() {
        scrollAmount += scrollStep;
        if (scrollAmount >= scrollingDiv.scrollWidth - scrollingDiv.clientWidth) {
            scrollAmount = 0; 
        } 
        scrollingDiv.scrollLeft = scrollAmount; 
        requestAnimationFrame(autoScroll); 
    } 
    autoScroll(); 
} 
createAutoScroll('scrolling-div', 1.5); 
createAutoScroll('scrolling-div1', 1.5); 
createAutoScroll('scrolling-div2', 1.5); 
createAutoScroll('scrolling-div4', 1.5); 
createAutoScroll('scrolling-div', 1.5); 
createAutoScroll('scrolling-divIL', 1.5); 
createAutoScroll('scrolling-divUS', 1.5); 
createAutoScroll('scrolling-divAU', 1.5); 
createAutoScroll('scrolling-divIT', 1.5); 
createAutoScroll('scrolling-divIN', 1.5); 
createAutoScroll('scrolling-divSP', 1.5); 
createAutoScroll('scrolling-divRU', 1.5); 
createAutoScroll('scrolling-divCN', 1.5); 
createAutoScroll('scrolling-divJA', 1.5); 
let trendingSwitch = document.getElementById("trending_switch");
let trendingChoice = document.getElementById("trending_choice");
let trendingWeekList = document.getElementById("trending_week_list");
let trendingDayList = document.getElementById("trending_day_list");
switch_trending();
function switch_trending(){
    if (trendingSwitch.checked === true) {
        trendingChoice.innerHTML = "Week";
        // fetch('.\templates\Trending_Week.html')
        //     .then(response => response.text())
        //     .then(data => {
        //         trendingList.innerHTML = data;
        //     })
        //     .catch(error => console.error('Error:', error));
        trendingWeekList.style.display="block";
        trendingDayList.style.display="none";
    } else {
        trendingChoice.innerHTML = "Day";
        trendingWeekList.style.display="none";
        trendingDayList.style.display="block";
        // fetch('Trending_Day.html')
        //     .then(response => response.text())
        //     .then(data => {
        //         trendingList.innerHTML = data;
        //     })
        //     .catch(error => console.error('Error:', error));
    }
    
    
}
radio_country();
document.getElementById("flexRadioIsrael").checked=true;
function radio_country() {
    const countries = ["IL", "AU", "US", "JA", "CN", "RU", "SP", "IN", "IT"];

    // Hide all scrolling-div elements
    countries.forEach(country => {
        document.getElementById(`${country}`).style.display = "none";
    });
    console.log("sfdd");
    // Check which radio button is checked and show the corresponding div
    switch (true) {
        case document.getElementById("flexRadioIsrael").checked:
            document.getElementById("IL").style.display = "block";
            break;
        case document.getElementById("flexRadioAustralia").checked:
            document.getElementById("AU").style.display = "block";
            break;
        case document.getElementById("flexRadioUSA").checked:
            document.getElementById("US").style.display = "block";
            break;
        case document.getElementById("flexRadioJapan").checked:
            document.getElementById("JA").style.display = "block";
            break;
        case document.getElementById("flexRadioRussia").checked:
            document.getElementById("RU").style.display = "block";
            break;
        case document.getElementById("flexRadioIndia").checked:
            document.getElementById("IN").style.display = "block";
            break;
        case document.getElementById("flexRadioSpain").checked:
            document.getElementById("SP").style.display = "block";
            break;
        case document.getElementById("flexRadioChina").checked:
            document.getElementById("CN").style.display = "block";
            break;
        case document.getElementById("flexRadioItaly").checked:
            document.getElementById("IT").style.display = "block";
            break;
    }
}

