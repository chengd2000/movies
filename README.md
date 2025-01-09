# Movie Project

## Overview
The Movie Project is a Python-based application that leverages Artificial Intelligence (AI) and data analytics to provide a powerful platform for managing and exploring movie data. This project includes an API that allows users to interact with the movie dataset programmatically.

## Features
- **Search and Filter Movies**: Perform advanced searches and filters based on:
  - Genre
  - Actor
  - Release Date
  - Rating
  - Movie Title
- **AI Integration**: Utilize AI algorithms for recommendations and insights based on movie data.
- **Data Analytics**: Gain insights from detailed movie analytics.
- **API Integration**: Use a RESTful API to interact with the system programmatically.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python run.py
   ```

## Usage

Once the application is running, you can:
- **Access the API**: Interact with endpoints to query and filter movies.
- **Use the Command Line Interface (CLI)**: Filter movies using the following criteria:
  - **Genre**: Find movies within a specific genre.
  - **Actor**: Locate movies featuring specific actors.
  - **Release Date**: Filter by release year or range.
  - **Rating**: Search for movies within a specific rating range.
  - **Title**: Search for movies by their title.

### Example Commands

- Filter movies by genre:
  ```bash
  python run.py --filter genre "Action"
  ```

- Search for movies by actor:
  ```bash
  python run.py --filter actor "Leonardo DiCaprio"
  ```

- Retrieve movies by rating range:
  ```bash
  python run.py --filter rating "8-10"
  ```

## Configuration

The project is configured using a settings file (`config.py`) which includes:
- API keys for external integrations.
- Database connection settings.
- AI model parameters.

Ensure the `config.py` file is properly set up before running the project.

## API Documentation

The API provides endpoints for querying movie data. Example:

- **GET /movies**
  Retrieve a list of movies with optional filters for genre, actor, release date, rating, or title.

- **GET /movies/<movie_id>**
  Retrieve detailed information about a specific movie.

Refer to the `api_docs.md` file for full API documentation.

## Dependencies

- **Python 3.8+**
- **Flask** (for API implementation)
- **Pandas** (for data analysis)
- **Scikit-learn** or similar library (for AI capabilities)
- **Requests** (for making external API calls, if needed)
