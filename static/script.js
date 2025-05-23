document.addEventListener("DOMContentLoaded", () => {

  const searchInput = document.getElementById("search-input")
  const searchButton = document.getElementById("search-button")
  const difficultyFilter = document.getElementById("difficulty-filter")
  // const ratingFilter = document.getElementById("rating-filter")
  const ratingSlider = document.getElementById('rating-filter');
  const ratingValue = document.getElementById('rating-value');
  const resultsSection = document.getElementById("results-section")
  const noResultsSection = document.getElementById("no-results")
  const courseGrid = document.getElementById("course-grid")
  const resultsCount = document.getElementById("results-count")
  const courseCardTemplate = document.getElementById("course-card-template")

  let isSearching = false

  searchButton.addEventListener("click", (e) => {
    e.preventDefault() 
    handleSearch()
  })
  
  searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault() 
      handleSearch()
    }
  })
  
  let filterTimeoutId = null
  difficultyFilter.addEventListener("change", (e) => {
    e.preventDefault()
    clearTimeout(filterTimeoutId)
    filterTimeoutId = setTimeout(handleSearch, 300)
  })

  // ratingFilter.addEventListener("change", (e) => {
  //   e.preventDefault()
  //   clearTimeout(filterTimeoutId)
  //   filterTimeoutId = setTimeout(handleSearch, 300)
  // })
  
  ratingSlider.addEventListener("change", (e) => {
    e.preventDefault()
    clearTimeout(filterTimeoutId)
    filterTimeoutId = setTimeout(handleSearch, 300)
  })

  if (ratingSlider && ratingValue) {
    ratingValue.textContent = Number(ratingSlider.value).toFixed(1);
    ratingSlider.addEventListener('input', function() {
      ratingValue.textContent = Number(ratingSlider.value).toFixed(1);
    });
  }
  
  noResultsSection.classList.add("hidden")
  resultsSection.classList.add("hidden")

  async function handleSearch() {

    if (isSearching) return
    isSearching = true
    
    const searchQuery = searchInput.value.trim()
    const difficultyValue = difficultyFilter.value
    const ratingValue = ratingSlider.value
    // const ratingValue = ratingFilter.value

    try {
      courseGrid.innerHTML = '<div class="loading">Loading results...</div>'
      resultsSection.classList.remove("hidden")
      noResultsSection.classList.add("hidden")

      // make API request to Flask
      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          difficulty: difficultyValue,
          rating: ratingValue
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const courses = await response.json()
      // check if courses is valid before displaying
      if (Array.isArray(courses)) {
        displayResults(courses)
      } else {
        throw new Error('Invalid response format')
      }
    } catch (error) {
      console.error('Error fetching course recommendations:', error)
      noResultsSection.classList.remove("hidden")
      resultsSection.classList.add("hidden")
      document.querySelector("#no-results p").textContent = "An error occurred while fetching recommendations. Please try again."
    } finally {
      isSearching = false
    }
  }

  function displayResults(courses) {
    // clear previous results
    courseGrid.innerHTML = ""

    if (courses.length === 0) {
      resultsSection.classList.add("hidden")
      noResultsSection.classList.remove("hidden")
      return
    }

    // update results count
    resultsCount.textContent = `Found ${courses.length} course${courses.length === 1 ? "" : "s"}`

    // show results section
    resultsSection.classList.remove("hidden")
    noResultsSection.classList.add("hidden")

    // create and append course cards
    courses.forEach((course) => {
      const courseCard = createCourseCard(course)
      courseGrid.appendChild(courseCard)
    })
  }

  function createCourseCard(course) {
    const cardClone = document.importNode(courseCardTemplate.content, true)
    const card = cardClone.querySelector(".course-card")

    // clean text
    const cleanText = (text) => {
      if (!text) return "";
      return text.replace(/\?/g, "'").replace(/\\'/g, "'").replace(/\\/g, "");
    }

    // set course details with clean text
    card.querySelector(".course-title").textContent = cleanText(course["Course Name"])
    card.querySelector(".university").textContent = cleanText(course["University"])
    card.querySelector(".rating-value").textContent = course["Course Rating"]

    // set difficulty
    const difficultyElement = card.querySelector(".difficulty")
    difficultyElement.textContent = course["Difficulty Level"]
    difficultyElement.classList.add(course["Difficulty Level"].toLowerCase())

    // set description
    card.querySelector(".course-description").textContent = cleanText(course["Course Description"])

    // set course URL
    const courseLink = card.querySelector(".course-link")
    courseLink.href = course["Course URL"]

    // create rating stars
    const ratingStars = card.querySelector(".rating-stars")
    const rating = Number.parseFloat(course["Course Rating"])
    ratingStars.innerHTML = generateStars(rating)

    // add skills max 10
    const skillsList = card.querySelector(".skills-list")
    let skills = []
    
    if (course["Skills"]) {
      skills = course["Skills"].split(",")
        .map(skill => skill.trim())
        .filter(skill => skill.length > 0)
        .slice(0, 10) 
    }
    
    skills.forEach((skill) => {
      const skillTag = document.createElement("span")
      skillTag.className = "skill-tag"
      skillTag.textContent = cleanText(skill)
      skillsList.appendChild(skillTag)
    })

    // Add keywords if available - limited to max 10
    // const keywordsList = card.querySelector(".keywords-list")
    // if (course["Keywords"]) {
    //   const keywords = course["Keywords"].split(/\s+/)
    //     .map(keyword => keyword.trim())
    //     .filter(keyword => keyword.length > 0)
    //     .slice(0, 10) // Limit to max 10 keywords
      
    //   keywords.forEach((keyword) => {
    //     if (keyword) {
    //       const keywordTag = document.createElement("span")
    //       keywordTag.className = "keyword-tag"
    //       keywordTag.textContent = cleanText(keyword)
    //       keywordsList.appendChild(keywordTag)
    //     }
    //   })
    // }

    return card
  }

  function generateStars(rating) {
    const fullStars = Math.floor(rating)
    const hasHalfStar = rating % 1 >= 0.5
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0)

    let starsHTML = ""

    // full 
    for (let i = 0; i < fullStars; i++) {
      starsHTML += "★"
    }

    // half
    if (hasHalfStar) {
      starsHTML += "★"
    }

    // empty
    for (let i = 0; i < emptyStars; i++) {
      starsHTML += "☆"
    }

    return starsHTML
  }
})