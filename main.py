import openai
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI()

# Define the schema for user input
class UserInput(BaseModel):
    answer: str
    session_id: Optional[str] = None

# Store active user sessions (mocked in memory)
sessions = {}

# Expected answer types for validation
expected_answers = {
    "state": "Any valid US state name (full name or abbreviation)",
    "dob": "Date of birth in any valid format",
    "emergency": "YES or NO(indicating an emergency)",
    "suicide_risk_1": "YES or NO(indicating current suicidal thoughts)",
    "suicide_risk_2": "YES or NO(indicating specific plan)",
    "suicide_risk_3": "YES or NO(indicating recent attempts)",
    "consent": "Yes, Accept, or similar affirmative response",
    "complaint": "Any description of mental health concerns",
    "goal": "Any reasonable treatment goal",
    "symptoms_start": "Time indication (days, weeks, months, years ago)",
    "symptoms_severity": "Number between 0-10 or descriptive severity",
    "symptom_triggers": "Any factors that worsen or better symptoms",
    "symptom_impact": "How it affects you",
    "past_episodes": "Yes/No response about previous experiences",
    "mental_health_history": "Yes/No response about prior diagnoses",
    "hospitalizations": "Yes/No response about hospitalizations"
}

# Question list to iterate through
questions = [
    ("What US state do you reside in?", "state"),
    ("What is your date of birth?", "dob"),
    ("Is this an emergency or are you in immediate danger? [Y/N]", "emergency"),
    ("Are you having thoughts of killing yourself right now?", "suicide_risk_1"),
    ("Do you have a specific plan or the means to do so?", "suicide_risk_2"),
    ("Have you attempted suicide in the past month?", "suicide_risk_3"),
    ("Do you agree to an AI-assisted intake and to audio recording/transcription for documentation?", "consent"),
    ("What brings you in today?", "complaint"),
    ("What's your top goal for today's visit?", "goal"),
    ("When did these concerns start?", "symptoms_start"),
    ("How severe are your symptoms today (0–10)?", "symptoms_severity"),
    ("What makes them better or worse?", "symptom_triggers"),
    ("How do they affect work/school, relationships, sleep, appetite?", "symptom_impact"),
    ("Have you had similar episodes before? What helped then?", "past_episodes"),
    ("Have you been given any mental-health diagnoses before?", "mental_health_history"),
    ("Any hospitalizations, ER visits, or suicide attempts?", "hospitalizations")
]

async def detect_suicide_risk(user_response: str, question_key: str = None) -> bool:
    """Detect if user response indicates suicide risk requiring immediate intervention"""
    
    # Define suicide-related question keys
    suicide_risk_questions = ["suicide_risk_1", "suicide_risk_2", "suicide_risk_3", "emergency"]
    
    try:
        # Check if this is a suicide-related question with positive response
        if question_key in suicide_risk_questions:
            prompt = f"""
            Question Key: {question_key}
            User Response: "{user_response}"
            
            Does this response indicate YES/positive to suicide risk or emergency?
            Look for affirmative responses like: yes, y, yeah, true, I am, I do, I have, definitely, absolutely, etc.
            
            Respond with "RISK" if positive/yes response detected, "SAFE" if negative/no response.
            """
        else:
            # Check any response for suicide-related content
            prompt = f"""
            User Response: "{user_response}"
            
            Does this response contain any indication of:
            - Current suicidal thoughts or ideation
            - Plans to harm themselves
            - Recent suicide attempts
            - Immediate danger or emergency situation
            - Expressions of wanting to die or kill themselves
            - Intent to commit suicide
            
            Respond with "RISK" if suicide risk detected, "SAFE" if no risk indicators found.
            """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip().upper()
        print(f"Suicide risk check for '{question_key}': {result}")
        return "RISK" in result
        
    except Exception as e:
        print(f"Error in suicide risk detection: {e}")
        # Err on the side of caution - if we can't check, assume risk for suicide-specific questions
        if question_key in suicide_risk_questions:
            return True
        return False

def get_emergency_message():
    """Return emergency intervention message"""
    return {
        "EMERGENCY": True,
        "message": "🚨 IMMEDIATE SAFETY CONCERN DETECTED 🚨",
        "emergency_resources": {
                "suicide_prevention": "National Suicide Prevention Lifeline: 988 or 1-800-273-8255",
        },
        "survey_status": "Survey has been paused for your safety. Please seek immediate help."
    }

async def validate_current_answer_from_multi_response(user_response: str, current_question: str, current_key: str):
    """Extract and validate just the current question's answer from a multi-answer response"""
    try:
        prompt = f"""
        Current Question: {current_question}
        User's Full Response: "{user_response}"
        
        Extract ONLY the part of the user's response that answers the current question.
        If the response contains multiple answers, isolate just the relevant part.
        
        Return only the extracted answer for the current question, nothing else.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        current_answer = response.choices[0].message.content.strip()
        print(f"Current answer extracted for {current_key}: {current_answer}")
        return current_answer
        
    except Exception as e:
        print(f"Error extracting current answer: {e}")
        return user_response  # Fallback to full response

async def extract_and_validate_multiple_answers(user_response: str, current_question_key: str, current_question_index: int) -> Dict[str, str]:
    """Extract multiple answers and validate each one individually"""
    try:
        # Get future questions only
        future_questions = []
        for i, (question, key) in enumerate(questions):
            if i > current_question_index:  # Only future questions
                future_questions.append(f"{key}: {question}")
        
        if not future_questions:
            return {}
        
        extraction_prompt = f"""
        User Response: "{user_response}"
        Current Question Key: {current_question_key}
        
        Future Questions:
        {chr(10).join(future_questions)}
        
        Extract answers for future questions that the user mentioned in their response.
        For each answer found, provide the exact text that answers that specific question.
        
        Return JSON format:
        {{"question_key": "exact_answer_text"}}
        
        If no future answers found, return: {{}}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            max_tokens=300,
            temperature=0.2
        )
        
        extracted_text = response.choices[0].message.content.strip()
        print(f"Raw extraction result: {extracted_text}")
        
        try:
            extracted_answers = json.loads(extracted_text)
            if not isinstance(extracted_answers, dict):
                return {}
        except json.JSONDecodeError:
            print("Failed to parse JSON from extraction")
            return {}
        
        # Now validate each extracted answer individually
        validated_answers = {}
        
        for question_key, extracted_answer in extracted_answers.items():
            # Find the full question text
            question_text = None
            for q_text, q_key in questions:
                if q_key == question_key:
                    question_text = q_text
                    break
            
            if question_text:
                # Validate this specific answer
                validation_prompt = f"""
                Question: {question_text}
                Extracted Answer: {extracted_answer}
                
                Is this extracted answer appropriate and sufficient for this specific question?
                Consider if the answer directly addresses what the question is asking.
                Respond with "Yes" or "No".
                """
                
                validation_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": validation_prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                
                is_valid = 'yes' in validation_response.choices[0].message.content.lower()
                
                if is_valid:
                    validated_answers[question_key] = extracted_answer
                    print(f"✓ Validated answer for {question_key}: {extracted_answer}")
                else:
                    print(f"✗ Invalid answer for {question_key}: {extracted_answer}")
        
        return validated_answers
        
    except Exception as e:
        print(f"Error in extract_and_validate_multiple_answers: {e}")
        return {}

async def verify_answer(question: str, user_answer: str, expected_type: str):
    """Verify if an answer is appropriate for the question"""
    try:
        prompt = f"""
        Question: {question}
        User's Answer: {user_answer}
        Expected Type: {expected_type}
        
        Is this answer appropriate and sufficient for the question? 
        Be flexible with reasonable variations and formats.
        Respond with "Yes" if appropriate, "No" if clearly inappropriate or insufficient.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        
        print(f"Validation result: {response.choices[0].message.content}")
        gpt_answer = response.choices[0].message.content.strip().lower()
        return 'yes' in gpt_answer

    except Exception as e:
        print(f"Error verifying answer: {e}")
        return True  # Accept answer if API fails

async def get_next_question(session):
    """Get the next question, handling pre-extracted answers"""
    current_index = session["question_index"]
    
    if current_index >= len(questions):
        return {"message": "Survey completed successfully!", "answers": session["answers"]}
    
    question, key = questions[current_index]
    
    # Check if we have a pre-extracted answer
    if key in session["extracted_answers"]:
        extracted_answer = session["extracted_answers"][key]
        session["confirmation_pending"] = key
        return {
            "message": f"I found that you previously mentioned: '{extracted_answer}' for the question '{question}'. Is this correct?",
            "options": "Reply 'Yes' to confirm or provide a new answer."
        }
    
    return {"question": question}

# Combined API for starting and continuing the survey
@app.post("/survey/")
async def survey(user_input: UserInput):
    # Initialize new session
    if not user_input.session_id or user_input.session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "question_index": 0,
            "answers": {},
            "extracted_answers": {},  # Store extracted future answers
            "confirmation_pending": None,  # Track if waiting for confirmation
            "greeting_done": False,
            "emergency_triggered": False
        }
        return {"session_id": session_id, "message": "Hi! Let's start with some questions. Please respond with 'Hi' to begin."}

    session = sessions[user_input.session_id]
    current_question_index = session["question_index"]
    
    # Check if emergency was already triggered
    if session.get("emergency_triggered", False):
        return get_emergency_message()
    
    # Handle greeting
    if not session["greeting_done"]:
        session["greeting_done"] = True
        return {"message": "Hi! Let's start with some questions. Please respond with 'Hi' to begin."}

    # Start survey
    if user_input.answer.lower() == "hi":
        question, key = questions[0]
        return {"question": question}

    # GENERAL SUICIDE RISK CHECK (after greeting is done)
    if session.get("greeting_done", False):
        general_risk_check = await detect_suicide_risk(user_input.answer)
        if general_risk_check:
            session["emergency_triggered"] = True
            return get_emergency_message()

    # Check if we're waiting for confirmation
    if session.get("confirmation_pending"):
        confirmation_key = session["confirmation_pending"]
        if user_input.answer.lower() in ["yes", "y", "correct", "that's right", "right"]:
            # User confirmed the extracted answer
            question, key = questions[current_question_index]
            confirmed_answer = session["extracted_answers"][key]
            
            # SUICIDE RISK CHECK FOR CONFIRMED ANSWER
            suicide_risk_detected = await detect_suicide_risk(confirmed_answer, key)
            if suicide_risk_detected:
                session["answers"][key] = confirmed_answer
                session["emergency_triggered"] = True
                return get_emergency_message()
            
            session["answers"][key] = session["extracted_answers"][key]
            del session["extracted_answers"][confirmation_key]  # Remove from extracted since it's now confirmed
            session["confirmation_pending"] = None
            session["question_index"] += 1
            
            # Move to next question
            if session["question_index"] < len(questions):
                return await get_next_question(session)
            else:
                return {"message": "Survey completed successfully!", "answers": session["answers"]}
        else:
            # User wants to provide new answer or gave a different answer
            session["confirmation_pending"] = None
            # Remove the rejected extracted answer
            if confirmation_key in session["extracted_answers"]:
                del session["extracted_answers"][confirmation_key]
            
            question, key = questions[current_question_index]
            
            # Validate the new answer they provided
            is_valid = await verify_answer(question, user_input.answer, expected_answers[key])
            
            if not is_valid:
                return {"message": f"The answer for '{question}' seems inappropriate. Please provide a clearer answer."}
            
            # SUICIDE RISK CHECK FOR NEW ANSWER
            suicide_risk_detected = await detect_suicide_risk(user_input.answer, key)
            if suicide_risk_detected:
                session["answers"][key] = user_input.answer
                session["emergency_triggered"] = True
                return get_emergency_message()
            
            # Save the new answer and continue
            session["answers"][key] = user_input.answer
            session["question_index"] += 1
            
            # Extract any additional answers from this response
            validated_future_answers = await extract_and_validate_multiple_answers(
                user_input.answer, key, current_question_index
            )
            if validated_future_answers:
                # CHECK EXTRACTED ANSWERS FOR SUICIDE RISK
                for future_key, future_answer in validated_future_answers.items():
                    suicide_risk_in_future = await detect_suicide_risk(future_answer, future_key)
                    if suicide_risk_in_future:
                        session["emergency_triggered"] = True
                        return get_emergency_message()
                
                session["extracted_answers"].update(validated_future_answers)
                print(f"✓ Extracted {len(validated_future_answers)} additional answers")
            
            if session["question_index"] < len(questions):
                return await get_next_question(session)
            else:
                return {"message": "Survey completed successfully!", "answers": session["answers"]}

    # Survey completed check
    if current_question_index >= len(questions):
        return {"message": "Survey completed successfully!", "answers": session["answers"]}

    # Get current question
    question, key = questions[current_question_index]
    
    # Check for pre-extracted answer
    if key in session["extracted_answers"]:
        extracted_answer = session["extracted_answers"][key]
        session["confirmation_pending"] = key
        return {
            "message": f"I found that you previously mentioned: '{extracted_answer}' for the question '{question}'. Is this correct?",
            "options": "Reply 'Yes' to confirm or provide a new answer."
        }
    
    # Handle current answer - extract it from potentially multi-answer response
    current_answer = await validate_current_answer_from_multi_response(
        user_input.answer, question, key
    )
    
    # Validate the current answer
    is_current_valid = await verify_answer(question, current_answer, expected_answers[key])
    
    if not is_current_valid:
        return {
            "message": f"The answer for '{question}' seems inappropriate. Please provide a clearer answer for this specific question.",
            "question": question
        }

    # SUICIDE RISK CHECK FOR CURRENT ANSWER
    suicide_risk_detected = await detect_suicide_risk(current_answer, key)
    if suicide_risk_detected:
        session["answers"][key] = current_answer
        session["emergency_triggered"] = True
        return get_emergency_message()

    # Save current answer
    session["answers"][key] = current_answer

    # Extract and validate future answers from the full response
    validated_future_answers = await extract_and_validate_multiple_answers(
        user_input.answer, key, current_question_index
    )
    
    if validated_future_answers:
        # CHECK EXTRACTED ANSWERS FOR SUICIDE RISK
        for future_key, future_answer in validated_future_answers.items():
            suicide_risk_in_future = await detect_suicide_risk(future_answer, future_key)
            if suicide_risk_in_future:
                session["emergency_triggered"] = True
                return get_emergency_message()
        
        session["extracted_answers"].update(validated_future_answers)
        extracted_count = len(validated_future_answers)
        print(f"✓ Successfully extracted and validated {extracted_count} future answers")

    # Move to next question
    session["question_index"] += 1
    if session["question_index"] < len(questions):
        return await get_next_question(session)
    else:
        return {"message": "Survey completed successfully!", "answers": session["answers"]}

# Optional: Add endpoint to view session details (for debugging)
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id in sessions:
        return sessions[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

