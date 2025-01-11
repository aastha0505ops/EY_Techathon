import re
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File,Form
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import io

app = FastAPI()

# Model initialization
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')


# Data models
class User(BaseModel):
    email: EmailStr
    password: str
    full_name: str


class SkillAssessment(BaseModel):
    skills: List[str]
    proficiency_levels: List[int]


class CourseRecommendation(BaseModel):
    course_name: str
    description: str
    skill_level: str


class CareerPath(BaseModel):
    path_name: str
    description: str
    required_skills: List[str]


class Progress(BaseModel):
    completion_percentage: float
    last_activity: str
    feedback: Optional[str] = None

# File paths
USERS_FILE = "data/users.json"
ASSESSMENTS_FILE = "data/assessments.json"
RECOMMENDATIONS_FILE = "data/recommendations.json"
CAREERS_FILE = "data/careers.json"
PROGRESS_FILE = "data/progress.json"


# Helper functions
def load_json(file_path: str) -> dict:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}


def save_json(file_path: str, data: dict):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def generate_response(prompt, system_message, max_new_tokens=1192):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def extract_text_from_resume(file_path) -> str:
    """Extract text from a PDF resume."""
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def analyze_skills(career_goal:str , resume_text: str) -> list[dict[Any, Any] | Any]:
    """Analyze skills from resume text using the LLM."""
    system_message = """You are a skilled HR professional who specializes in analyzing resumes and identifying technical skills. 
    Extract technical skills and estimate proficiency level based on the context, experience, and projects mentioned. 
    Return the response as a summary for all the skills the user has"""

    prompt = f"Analyze the following resume and list all technical skills with their proficiency level:\n\n{resume_text} and give the response in JSON format ()"

    response = generate_response(prompt, system_message)
    # Regular expression to extract the JSON content
    pattern = r'```json\n(.*?)\n```'

    # Extracting JSON content using re.DOTALL to handle multi-line JSON
    matches = re.findall(pattern, response, re.DOTALL)
    parsed_json = {}
    if matches:
        json_content = matches[0].strip()
        print("Extracted JSON Content:")
        print(json_content)

        # Optional: Parse the JSON string into a Python dictionary
        try:
            parsed_json = json.loads(json_content)
            print("\nParsed JSON Data:")
            print(json.dumps(parsed_json, indent=2))
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
    else:
        print("No JSON content found.")

    system_message = f"""You are a ATS system based on the resume data give me a summary for the candidate and give skill match the skill gaps for {career_goal} in the candidate. give output in paragraph"""

    prompt = f"give me a summary for the candidate and give the skill match and skill gaps for {career_goal} in the candidate given resume analysis:\n\n{resume_text}"

    summary = generate_response(prompt, system_message)
    print(summary)

    system_message = f"""You are a ATS system based on the resume data give me a score for the candidate for {career_goal}. give output in JSON"""

    prompt = f"give me a single candidate Score based on skill match and skill gaps for {career_goal} in the candidate summary:\n\n{summary}, in JSON for example {{Score:9}}"

    resume_score = generate_response(prompt, system_message,max_new_tokens=30)

    print(resume_score)

    pattern = r'```json\n(.*?)\n```'

    # Extracting JSON content using re.DOTALL to handle multi-line JSON
    matches = re.findall(pattern, resume_score, re.DOTALL)
    scores_json = {}
    if matches:
        json_content = matches[0].strip()
        print(json_content)

        # Optional: Parse the JSON string into a Python dictionary
        try:
            scores_json = json.loads(json_content)
            print("\nParsed JSON Data:",scores_json)
            # print(json.dumps(parsed_json, indent=2))
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)

    return [parsed_json, summary, scores_json.get("Score","")]


# API endpoints
@app.post("/register")
async def register(user: User):
    users = load_json(USERS_FILE)

    if user.email in users:
        raise HTTPException(status_code=400, detail="Email already registered")

    users[user.email] = {
        "password": user.password,  # In production, hash this!
        "full_name": user.full_name
    }

    save_json(USERS_FILE, users)
    return {"message": "Registration successful"}


@app.post("/login")
async def login(email: str, password: str):
    users = load_json(USERS_FILE)

    if email not in users or users[email]["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": f"Welcome {users[email]['full_name']}!"}


@app.post("/upload-resume")
async def upload_resume(email: str, career_goal: str , file: UploadFile = File(...)):
    """Upload and analyze resume for automated skill assessment."""
    # Verify user exists
    users = load_json(USERS_FILE)
    if email not in users:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # Read and extract text from resume
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            resume_path = temp_file.name

        resume_text = extract_text_from_resume(resume_path)

        print(resume_text)

        # Analyze skills using LLM
        skills_assesment, summary, resume_score = analyze_skills(career_goal,resume_text)

        # Save assessment
        assessment = {
            "summary": summary,
            "skills_assesment":skills_assesment,
            "resume_score" : resume_score
        }

        assessments = load_json(ASSESSMENTS_FILE)
        assessments[email] = assessment
        save_json(ASSESSMENTS_FILE, assessments)

        return {
            "message": "Resume analyzed successfully",
            "summary" : summary,
            "resume_score": resume_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")


# Rest of the endpoints remain the same...
@app.get("/recommendations/{email}")
async def get_recommendations(email: str):
    assessments = load_json(ASSESSMENTS_FILE)
    if email not in assessments:
        raise HTTPException(status_code=400, detail="Please complete skill assessment first")

    recommendations = []
    for skill, level in zip(assessments[email]["skills"],
                            assessments[email]["proficiency_levels"]):
        recommendations.append({
            "course_name": f"Advanced {skill}",
            "description": f"Improve your {skill} skills",
            "skill_level": "Intermediate" if level < 7 else "Advanced"
        })

    recommendations_db = load_json(RECOMMENDATIONS_FILE)
    recommendations_db[email] = recommendations
    save_json(RECOMMENDATIONS_FILE, recommendations_db)

    return recommendations


@app.get("/career-paths/{email}")
async def get_career_paths(email: str):
    assessments = load_json(ASSESSMENTS_FILE)
    if email not in assessments:
        raise HTTPException(status_code=400, detail="Please complete skill assessment first")

    career_paths = []
    for skill in assessments[email]["skills"]:
        career_paths.append({
            "path_name": f"{skill} Specialist",
            "description": f"Become a professional {skill} specialist",
            "required_skills": [skill, "Communication", "Problem Solving"]
        })

    careers_db = load_json(CAREERS_FILE)
    careers_db[email] = career_paths
    save_json(CAREERS_FILE, careers_db)

    return career_paths


@app.post("/progress/{email}")
async def update_progress(email: str, progress: Progress):
    progress_db = load_json(PROGRESS_FILE)
    progress_db[email] = {
        "completion_percentage": progress.completion_percentage,
        "last_activity": progress.last_activity,
        "feedback": progress.feedback
    }
    save_json(PROGRESS_FILE, progress_db)
    return {"message": "Progress updated successfully"}



def chat_bot(chat_history:str, max_new_tokens=600, max_history_length=5):
    """
    Generates a response using a preloaded model and tokenizer, maintaining a chat history.

    Args:
        user_input (str): The user's input message.
        max_new_tokens (int): The maximum number of tokens to generate. Default is 512.
        max_history_length (int): The maximum number of messages to retain in the chat history. Default is 5.

    Returns:
        str: The generated response from the model.
    """

    # Add the new user message to the chat history
    # chat_history.append({"role": "user", "content": user_input})
    chat_history = json.loads(chat_history)
    # Ensure the chat history does not exceed the maximum length
    if len(chat_history) > max_history_length:
        chat_history.pop(1)  # Remove the second message (first user message) to maintain context

    # Tokenize the messages using the chat template
    text = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Add the model's response to the chat history
    # chat_history.append({"role": "assistant", "content": response})

    return response

@app.post("/chat")
async def chat(chat_history:str):

    response = chat_bot(chat_history)

    return response