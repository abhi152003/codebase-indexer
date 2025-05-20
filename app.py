from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import git
import logging
import asyncio
import nest_asyncio
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import shutil
import requests
import uuid
import time
from datetime import datetime

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
    allow_origins=["http://localhost:3000", "http://localhost", "http://0.0.0.0:3000", "https://speedrunstylus.com", "http://speedrunstylus.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and tokenizer for CodeBERTa
MODEL_NAME = 'huggingface/CodeBERTa-small-v1'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# In-memory storage for tracking processing tasks
processing_tasks = {}

class RepoRequest(BaseModel):
    repo_url: str
    folders: List[str] = []

class BackgroundProcessRequest(BaseModel):
    repo_url: str
    description: str
    challenge_name: str
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

async def process_repo(processing_id: str, repo_url: str, description: str, challenge_name: str, folders: List[str]):
    """Background processing task to evaluate a repository and generate a score."""
    try:
        # Update task status to processing
        processing_tasks[processing_id]['status'] = 'processing'
        processing_tasks[processing_id]['start_time'] = time.time()
        
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_dir = f'repos/{repo_name}'
        
        # Clone the repository
        await clone_repo(repo_url, repo_dir)
        
        # Get code files from specified folders or entire repo
        code_files = get_code_files(repo_dir, folders)
        if not code_files:
            processing_tasks[processing_id]['status'] = 'failed'
            processing_tasks[processing_id]['error'] = 'No code files found in the specified folders or repository.'
            delete_repo(repo_dir)
            return
        
        # Process files in batches to improve performance
        BATCH_SIZE = 5  # Process 5 files at a time
        results = []
        
        processing_tasks[processing_id]['total_files'] = len(code_files)
        processing_tasks[processing_id]['processed_files'] = 0
        
        for i in range(0, len(code_files), BATCH_SIZE):
            batch = code_files[i:i+BATCH_SIZE]
            batch_tasks = [process_file(file_path, repo_url) for file_path in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Update progress
            processing_tasks[processing_id]['processed_files'] += len(batch)
            
            # Calculate estimated time remaining
            elapsed_time = time.time() - processing_tasks[processing_id]['start_time']
            files_processed = processing_tasks[processing_id]['processed_files']
            if files_processed > 0:
                time_per_file = elapsed_time / files_processed
                remaining_files = len(code_files) - files_processed
                est_time_remaining = time_per_file * remaining_files
                processing_tasks[processing_id]['estimatedTimeRemaining'] = int(est_time_remaining)
        
        success_count = sum(1 for result in results if result['status'] == 'success')
        error_count = len(results) - success_count
        
        # After embedding generation, query the code for review
        if success_count > 0:
            # Query codebase for evaluation
            query_result = await query_codebase_for_evaluation(repo_url, description, challenge_name)
            
            # Extract score from response
            score = extract_score_from_response(query_result.get('answer', ''))
            
            # Update task with result
            processing_tasks[processing_id]['status'] = 'completed'
            processing_tasks[processing_id]['result'] = {
                'answer': query_result.get('answer', ''),
                'score': score,
                'relevant_files': query_result.get('relevant_files', [])
            }
        else:
            processing_tasks[processing_id]['status'] = 'failed'
            processing_tasks[processing_id]['error'] = 'Failed to process any files successfully.'
        
        # Delete the repository after processing
        delete_repo(repo_dir)
        
    except Exception as e:
        logger.error(f'Error in background processing: {str(e)}')
        processing_tasks[processing_id]['status'] = 'failed'
        processing_tasks[processing_id]['error'] = str(e)
        delete_repo(repo_dir)

async def query_codebase_for_evaluation(repo_url: str, description: str, challenge_name: str) -> Dict:
    """Query the codebase using OpenAI with context from Pinecone embeddings."""
    try:
        # Generate embedding for the query
        query = f"{description} - Evaluate this {challenge_name} solution for code quality, best practices, and potential improvements."
        query_embedding = generate_embedding(query)
        
        # Query Pinecone for relevant code snippets
        query_result = index.query(
            vector=query_embedding.tolist(),
            top_k=2,
            include_metadata=True,
            filter={'repo_url': repo_url}
        )
        
        # Prepare context by fetching full content from GitHub
        context = []
        for match in query_result['matches']:
            file_path = match['metadata']['file_path']
            full_content = await fetch_file_from_github(repo_url, file_path)
            context.append(f'File: {file_path}\nContent:\n{full_content}\n')
        context_text = '\n\n'.join(context) if context else 'No relevant code snippets found.'
        
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

Student's description of changes: {description}

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
            'query': query,
            'answer': answer,
            'relevant_files': [match['metadata']['file_path'] for match in query_result['matches']]
        }
    except Exception as e:
        logger.error(f'Error querying codebase: {str(e)}')
        return {'error': str(e)}

def extract_score_from_response(answer: str) -> Optional[int]:
    """Extract a score from the model's response."""
    try:
        # Try to convert the answer directly to a number
        score = int(answer.strip())
        
        # Ensure the score is between 0 and 100
        return max(0, min(score, 100))
    except (ValueError, TypeError):
        logger.error(f"Failed to extract score from: {answer}")
        return None

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

@app.post('/begin-processing')
async def begin_processing(request: BackgroundProcessRequest, background_tasks: BackgroundTasks):
    """
    Start the background processing of a repository.
    Returns a processing ID that can be used to check the status.
    """
    processing_id = str(uuid.uuid4())
    
    # Store initial task data
    processing_tasks[processing_id] = {
        'status': 'submitted',
        'repo_url': request.repo_url,
        'description': request.description,
        'challenge_name': request.challenge_name,
        'folders': request.folders,
        'created_at': datetime.now().isoformat(),
        'estimatedTimeRemaining': 300  # Default 5 minutes estimate
    }
    
    # Start the background processing task
    background_tasks.add_task(
        process_repo,
        processing_id=processing_id,
        repo_url=request.repo_url,
        description=request.description,
        challenge_name=request.challenge_name,
        folders=request.folders
    )
    
    return {
        'processing_id': processing_id,
        'status': 'submitted',
        'estimatedTimeRemaining': 600  # Initial estimate of 10 minutes
    }

@app.get('/processing-status/{processing_id}')
async def check_processing_status(processing_id: str):
    """
    Check the status of a processing task.
    Returns the current status and any available results.
    """
    if processing_id not in processing_tasks:
        raise HTTPException(status_code=404, detail=f"Processing task {processing_id} not found")
    
    task = processing_tasks[processing_id]
    
    # Clean up completed tasks older than 1 hour
    current_time = time.time()
    for task_id in list(processing_tasks.keys()):
        task_data = processing_tasks[task_id]
        if 'start_time' in task_data and (task_data['status'] == 'completed' or task_data['status'] == 'failed'):
            if current_time - task_data.get('start_time', 0) > 3600:  # 1 hour
                del processing_tasks[task_id]
    
    return task

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8001) 