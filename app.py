from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import os
import git
import logging
import asyncio
import nest_asyncio
from pathlib import Path
from typing import List, Dict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import shutil
import requests

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to an index
INDEX_NAME = 'codebase-embeddings'
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENVIRONMENT if PINECONE_ENVIRONMENT else 'us-east-1'
        )
    )
index = pc.Index(INDEX_NAME)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio for running async code in non-async environments
nest_asyncio.apply()

app = FastAPI(title='Codebase Evaluation API')

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost", "http://0.0.0.0:3000", "https://speedrunstylus.com", "http://speedrunstylus.com"],  # Added your domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Model and tokenizer for CodeBERTa
MODEL_NAME = 'huggingface/CodeBERTa-small-v1'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Request timeout in seconds
REQUEST_TIMEOUT = 300  # 5 minutes timeout for long-running operations

class RepoRequest(BaseModel):
    repo_url: str
    folders: List[str] = []

class QueryRequest(BaseModel):
    repo_url: str
    query: str
    top_k: int = 2

async def clone_repo(repo_url: str, repo_dir: str) -> str:
    """Clone the repository to a local directory."""
    try:
        if os.path.exists(repo_dir):
            logger.info(f'Repository directory {repo_dir} already exists, skipping clone.')
            return repo_dir
        git.Repo.clone_from(repo_url, repo_dir)
        logger.info(f'Successfully cloned repository to {repo_dir}')
        return repo_dir
    except Exception as e:
        logger.error(f'Error cloning repository: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Error cloning repository: {str(e)}')

def get_code_files(repo_dir: str, folders: List[str]) -> List[str]:
    """Get list of code files from specified folders or entire repo."""
    code_extensions = ('.py', '.tsx', '.jsx', '.ts', '.js')
    code_files = []
    base_path = Path(repo_dir)
    search_paths = [base_path / folder for folder in folders] if folders else [base_path]
    
    for path in search_paths:
        if not path.exists():
            logger.warning(f'Folder not found: {path}')
            continue
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in code_extensions:
                code_files.append(str(file_path))
    return code_files

def generate_embedding(code_content: str) -> np.ndarray:
    """Generate embedding for the given code content using CodeBERTa."""
    try:
        inputs = tokenizer(code_content, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
    except Exception as e:
        logger.error(f'Error generating embedding: {str(e)}')
        return np.zeros(768)

async def process_file(file_path: str, repo_url: str) -> Dict:
    """Process a single file to generate and store its embedding."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        embedding = generate_embedding(content)
        file_id = f'{repo_url}/{file_path}'
        print("Embedding generated", embedding)
        print("Upserting...", embedding.tolist())
        index.upsert([(file_id, embedding.tolist(), {'file_path': file_path, 'repo_url': repo_url})])
        logger.info(f'Successfully processed and stored embedding for {file_path}')
        return {'file_path': file_path, 'status': 'success'}
    except Exception as e:
        logger.error(f'Error processing file {file_path}: {str(e)}')
        return {'file_path': file_path, 'status': 'error', 'error': str(e)}

def delete_repo(repo_dir: str) -> None:
    """Delete the cloned repository to save space."""
    try:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
            logger.info(f'Deleted repository directory {repo_dir}')
        else:
            logger.warning(f'Repository directory {repo_dir} not found for deletion')
    except Exception as e:
        logger.error(f'Error deleting repository {repo_dir}: {str(e)}')

@app.post('/evaluate-repo')
async def evaluate_repo(request: RepoRequest):
    """Endpoint to evaluate a GitHub repository by cloning it and storing embeddings of code files."""
    repo_url = request.repo_url
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_dir = f'repos/{repo_name}'
    
    try:
        # Set a timeout for the entire operation
        process_complete = False
        
        async def process_repo_with_timeout():
            nonlocal process_complete
            try:
                # Clone the repository
                await clone_repo(repo_url, repo_dir)
                
                # Get code files from specified folders or entire repo
                code_files = get_code_files(repo_dir, request.folders)
                if not code_files:
                    delete_repo(repo_dir)
                    raise HTTPException(status_code=404, detail='No code files found in the specified folders or repository.')
                
                # Process files in batches to improve performance
                BATCH_SIZE = 5  # Process 5 files at a time
                results = []
                
                for i in range(0, len(code_files), BATCH_SIZE):
                    batch = code_files[i:i+BATCH_SIZE]
                    batch_tasks = [process_file(file_path, repo_url) for file_path in batch]
                    batch_results = await asyncio.gather(*batch_tasks)
                    results.extend(batch_results)
                    # Log progress to monitor performance
                    logger.info(f'Processed batch {i//BATCH_SIZE + 1}/{(len(code_files) + BATCH_SIZE - 1)//BATCH_SIZE}')
                
                success_count = sum(1 for result in results if result['status'] == 'success')
                error_count = len(results) - success_count
                errors = [result for result in results if result['status'] == 'error']
                
                # Delete the repository after processing if there are successful embeddings
                if success_count > 0:
                    delete_repo(repo_dir)
                    
                process_complete = True
                return {
                    'message': f'Processed {len(code_files)} files. {success_count} successful, {error_count} failed.',
                    'processed_files': len(code_files),
                    'success_count': success_count,
                    'error_count': error_count,
                    'errors': errors
                }
            except Exception as e:
                delete_repo(repo_dir)
                logger.error(f'Error in process_repo_with_timeout: {str(e)}')
                raise e
        
        # Run the processing with a timeout
        try:
            result = await asyncio.wait_for(process_repo_with_timeout(), timeout=REQUEST_TIMEOUT)
            return result
        except asyncio.TimeoutError:
            logger.error(f'Repository processing timed out after {REQUEST_TIMEOUT} seconds')
            delete_repo(repo_dir)
            raise HTTPException(status_code=504, 
                detail=f'Repository processing timed out after {REQUEST_TIMEOUT} seconds. The repository may be too large or complex.')
        finally:
            # Clean up if not already done
            if not process_complete:
                delete_repo(repo_dir)
                
    except HTTPException as he:
        delete_repo(repo_dir)
        raise he
    except Exception as e:
        delete_repo(repo_dir)
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Unexpected error: {str(e)}')

@app.post('/query-codebase')
async def query_codebase(request: QueryRequest):
    """Endpoint to query the codebase using gpt-4o-mini with context from Pinecone embeddings."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail='OpenAI API key not configured.')
    
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(request.query)
        
        # Query Pinecone for relevant code snippets
        query_result = index.query(
            vector=query_embedding.tolist(),
            top_k=request.top_k,
            include_metadata=True,
            filter={'repo_url': request.repo_url}
        )
        
        print("query_result", query_result)
        
        # Prepare context by fetching full content from GitHub
        context = []
        for match in query_result['matches']:
            file_path = match['metadata']['file_path']
            full_content = await fetch_file_from_github(request.repo_url, file_path)
            context.append(f'File: {file_path}\nContent:\n{full_content}\n')
        context_text = '\n\n'.join(context) if context else 'No relevant code snippets found.'
        
        print("context_text", context_text)
        
        # Prepare the prompt for OpenAI
        prompt = f"""You are an objective code evaluator reviewing a student's submitted challenge. Based on the provided code context and the student's description of their changes, evaluate the code and return a single numeric score between 0-100.

Your evaluation must consider these specific metrics with equal weight (25% each):
1. Code quality (25 points): 
   - Clean, readable, and well-structured code
   - Proper variable/function naming
   - Appropriate comments and documentation
   - Modularity and organization
   - Avoidance of code duplication

2. Security considerations (25 points):
   - Protection against common vulnerabilities (e.g., injection, XSS)
   - Proper input validation and sanitization
   - Secure authentication and authorization (if applicable)
   - Safe handling of sensitive data
   - Avoidance of security anti-patterns

3. Uniqueness and creativity (25 points):
   - Innovative approach to problem-solving
   - Creative implementation of features
   - Original thinking in design decisions
   - Efficiency of the solution
   - Going beyond basic requirements

4. Adherence to best practices (25 points):
   - Following language-specific conventions
   - Using appropriate design patterns
   - Implementing error handling correctly
   - Writing testable code
   - Performance considerations

Code Context:
{context_text}

Student's description of changes: {request.query}

Your response must be ONLY a number between 0-100 representing the overall score. Do not include any explanations, comments, or additional text."""
        
        # Call OpenAI API
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'gpt-4o-mini',
            'messages': [
                {'role': 'system', 'content': 'You are an objective code evaluator for student challenges. You must respond ONLY with a single number between 0-100 representing your score based on code quality, security, creativity, and best practices. Do not include any text, explanations, or comments in your response.'},
                {'role': 'user', 'content': prompt}
            ],
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()
        answer = response.json()['choices'][0]['message']['content']
        
        return {
            'query': request.query,
            'answer': answer,
            'relevant_files': [match['metadata']['file_path'] for match in query_result['matches']]
        }
    except Exception as e:
        logger.error(f'Error querying codebase: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Error querying codebase: {str(e)}')

async def fetch_file_from_github(repo_url: str, file_path: str) -> str:
    """Fetch the full content of a file from GitHub using the GitHub API."""
    try:
        # Extract owner and repo name from the URL
        parts = repo_url.rstrip('/').split('/')
        owner = parts[-2]
        repo = parts[-1].replace('.git', '')
        
        # Extract the relative file path (remove the local repo_dir prefix)
        relative_path = file_path.split(f'repos/{repo}/')[-1] if f'repos/{repo}/' in file_path else file_path
        
        # GitHub API URL for file content
        github_api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{relative_path}'
        headers = {
            'Accept': 'application/vnd.github.v3.raw'
        }
        # If a GitHub token is provided, use it for higher rate limits (optional)
        github_token = os.getenv('GITHUB_TOKEN', '')
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        
        response = requests.get(github_api_url, headers=headers)
        if response.status_code == 200:
            logger.info(f'Successfully fetched content for {file_path} from GitHub')
            return response.text
        else:
            logger.error(f'Failed to fetch content for {file_path} from GitHub: {response.status_code}')
            return ''
    except Exception as e:
        logger.error(f'Error fetching file from GitHub for {file_path}: {str(e)}')
        return ''

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8001) 