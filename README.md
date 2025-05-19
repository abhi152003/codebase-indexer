# Codebase Evaluation API

This API allows users to submit a GitHub repository URL for evaluation. It clones the repository, generates embeddings for code files using the CodeBERTa-small-v1 model, and stores them in a Pinecone vector database for further analysis.

## Setup

1. **Install Dependencies**: Run `pip install -r requirements.txt` to install the required packages.
2. **Environment Variables**: Set up your Pinecone API key and environment in a `.env` file or directly in your environment.
   ```
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   ```
3. **Run the API**: Start the server with `python app.py` or use `uvicorn app:app --reload` for development mode.

## Usage

- **Endpoint**: `POST /evaluate-repo`
- **Request Body**:
  ```json
  {
    "repo_url": "https://github.com/user/repo.git",
    "folders": ["src", "lib"]
  }
  ```
- **Response**: The API will return a summary of processed files, including counts of successful and failed processing, along with any errors encountered.

## Notes

- The API currently supports a predefined set of code file extensions (e.g., `.py`, `.js`, `.java`).
- Embeddings are generated using the CodeBERTa-small-v1 model from Hugging Face.
- The Pinecone vector database is used to store embeddings for potential future similarity searches or analysis. 