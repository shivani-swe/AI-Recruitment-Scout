import os
from fastapi import FastAPI, UploadFile, File, Form
from pypdf import PdfReader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import io
from pydantic import BaseModel, Field
from typing import List
import asyncio
import docx2txt


load_dotenv()

app=FastAPI(title="Recursive Resume Scout")
llm= ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

#the RUBRIC
class CandidateEval(BaseModel):
    name: str= Field(description="Full Name of candidate")
    technical_score: int= Field(description="Score from 1-10 based on hard skills matching the JD")
    experience_score: int = Field(description="Score from 1-10 based on years and relevance of roles")
    missing_skills: List[str] = Field(description="List of key requirements from the JD not found in the resume")
    hire_recommendation: bool = Field(description="True if the candidate is a top-tier match")
    reasoning: str = Field(description="A 2-sentence justification for the scores")

class ExecutiveReport(BaseModel):
    pool_overview: str= Field(description="A summary of the total candidate pool quality")
    top_candidate_justification: str= Field(description="Detailed reason why the #1 person is the best fit")
    recommend_next_steps: str=Field(description="Advice for the recruiter on what to ask in interviews")


#ROUTER DOC OR PDF
def get_doc(file_bytes, filename):
    ext= filename.split('.')[-1].lower()

    if ext=="pdf":
        reader= PdfReader(io.BytesIO(file_bytes))
        return "".join([page.extract_text() for page in reader.pages])
    elif ext in ["docx", "doc"]:
        return docx2txt.process(io.BytesIO(file_bytes))
    else:
        return None

@app.post("/screen")
async def screen_candidate(jd: str= Form(...), file: UploadFile= File(...)):
    content= await file.read()

    resume_text= get_doc(content, file.filename)

    if not resume_text:
        return {"error": "Please upload a PDF or Word Document."}
    
    structured_llm= llm.with_structured_output(CandidateEval)

    prompt=f"""
    Evaluate the RESUME against the JD.
    JD: {jd}
    RESUME: {resume_text[:10000]}"""
    
    evaluation= structured_llm.invoke(prompt)
    return evaluation

@app.post("/screen-batch")
async def mupltiple_resume(jd: str= Form(...), files: List[UploadFile]= File(...)):
    all_evaluations= []

    for file in files:
        content= await file.read()
        resume_text= get_doc(content, file.filename)

        if not resume_text:
            continue

        structured_llm= llm.with_structured_output(CandidateEval)

        prompt=f"""
    Evaluate the following RESUME against the JOB DESCRIPTION(JD).
    JD: {jd}
    RESUME: {resume_text[:8000]}
    """
        
        evaluation= structured_llm.invoke(prompt)
        all_evaluations.append(evaluation)
    
    sorted_leaderboard= sorted(
        all_evaluations,
        key= lambda x: x.technical_score + x.experience_score,
        reverse=True 
    )

    return sorted_leaderboard

@app.post("/screen-batch-deep")

async def process_resume(file: UploadFile, jd: str):
    content= await file.read()
    resume_text= get_doc(content, file.filename)
    if not resume_text:            
        return None

    structured_llm= llm.with_structured_output(CandidateEval)
    prompt = f"Evaluate the RESUME against the JD.\nJD: {jd}\nRESUME: {resume_text[:8000]}"      
    return structured_llm.invoke(prompt)

async def deep_scout(jd: str= Form(...), files: List[UploadFile]= File(...)):

    tasks=[process_resume(f,jd) for f in files]
    results= await asyncio.gather(*tasks)
    individual_evals=[res for res in results if res is not None]

    if not individual_evals:
        return {"error": "No valid reumes processed."}
   
    sorted_list= sorted(individual_evals, key=lambda x: x.technical_score + x.experience_score, reverse=True)
    top_3= sorted_list[:3]

    analyse_llm= llm.with_structured_output(ExecutiveReport)

    analyse_prompt=f"""
    You are a Senior Hiring Partner. Review these TOP 3 candidate evaluations and write an Executive Report for the hiring manager.
    JOB DESCRIPTION: {jd}
    TOP CANDIDATES:
    {top_3}
    """
    executive_summary= analyse_llm.invoke(analyse_prompt)

    return {
        "leaderboard": sorted_list,
        "executive_report": executive_summary
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

