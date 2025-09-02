import openai
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

# Store active user sessions (mocked in memory)
sessions = {}

# Your existing expected_answers and questions remain the same
expected_answers = {
    "state": {
        "question": "What US state do you reside in?",
        "expected_format": "Valid US state name (full or abbreviation)",
        "examples": ["California", "CA", "New York", "NY"]
    },
    "dob": {
        "question": "What is your date of birth?",
        "expected_format": "Date of birth in valid format",
        "examples": ["01/23/1990", "1990-01-23", "23 Jan 1990"]
    },
    "emergency": {
        "question": "Is this an emergency or are you in immediate danger?",
        "expected_format": "YES or NO",
        "examples": ["YES", "NO"]
    },
    "suicide_risk_1": {
        "question": "Are you having thoughts of killing yourself right now?",
        "expected_format": "YES or NO",
        "examples": ["YES", "NO"]
    },
    "suicide_risk_2": {
        "question": "Do you have a specific plan or the means to do so?",
        "expected_format": "YES or NO",
        "examples": ["YES", "NO"]
    },
    "suicide_risk_3": {
        "question": "Have you attempted suicide in the past month?",
        "expected_format": "YES or NO",
        "examples": ["YES", "NO"]
    },
    "consent": {
        "question": "Do you agree to an AI-assisted intake and recording?",
        "expected_format": "Accept or Decline",
        "examples": ["Accept", "Yes", "Decline", "No"]
    },
    "complaint": {
        "question": "What brings you in today?",
        "expected_format": "Free text",
        "examples": ["I feel anxious all the time.", "I can't sleep well."]
    },
    "goal": {
        "question": "What's your top goal for today's visit?",
        "expected_format": "Free text",
        "examples": ["To manage my anxiety.", "To improve my sleep."]
    },
    "symptoms_start": {
        "question": "When did these concerns start?",
        "expected_format": "Time duration",
        "examples": ["2 weeks ago", "6 months ago", "years ago"]
    },
    "symptoms_severity": {
        "question": "How severe are your symptoms (0–10)?",
        "expected_format": "Number 0–10",
        "examples": ["3", "7", "10"]
    },
    "symptom_triggers": {
        "question": "What makes your symptoms better or worse?",
        "expected_format": "Free text",
        "examples": ["Stress makes it worse.", "Exercise helps."]
    },
    "symptom_impact": {
        "question": "How do symptoms affect work, school, sleep, relationships?",
        "expected_format": "Free text (related to how symptoms affects)",
        "examples": [
            "It affects my sleep.",
            "I can't focus in school.",
            "It causes problems at work."
        ]
    },
    "past_episodes": {
        "question": "Have you had similar episodes before?",
        "expected_format": "YES/NO + details",
        "examples": ["Yes, during college.", "No"]
    },
    "mental_health_history": {
        "question": "Have you been diagnosed with a mental health condition before?",
        "expected_format": "YES/NO + optional details",
        "examples": ["Yes, depression.", "No"]
    },
    "hospitalizations": {
        "question": "Any hospitalizations, ER visits, or suicide attempts?",
        "expected_format": "YES/NO + details",
        "examples": ["Yes, ER visit last year.", "No"]
    }
}

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
    ("How severe are your symptoms today(0-10)?", "symptoms_severity"),
    ("What makes them better or worse?", "symptom_triggers"),
    ("How do they affect work/school, relationships, sleep, appetite?", "symptom_impact"),
    ("Have you had similar episodes before? What helped then?", "past_episodes"),
    ("Have you been given any mental-health diagnoses before?", "mental_health_history"),
    ("Any hospitalizations, ER visits, or suicide attempts?", "hospitalizations")
]

# Keep all your existing helper functions with modified return format
async def detect_suicide_risk(user_response: str, question_key: str = None) -> bool:
    """Detect if user response indicates suicide risk requiring immediate intervention"""
    
    suicide_risk_questions = ["suicide_risk_1", "suicide_risk_2", "suicide_risk_3", "emergency"]
    
    try:
        if question_key in suicide_risk_questions:
            prompt = f"""
            Question Key: {question_key}
            User Response: "{user_response}"
            
            Does this response indicate YES/positive to suicide risk or emergency?
            Look for affirmative responses like: yes, y, yeah, true, I am, I do, I have, definitely, absolutely, etc.
            
            Respond with "RISK" if positive/yes response detected, "SAFE" if negative/no response.
            """
        else:
            prompt = f"""
            User Response: "{user_response}"
            
            Does this response contain any indication of:
            - Current suicidal thoughts.
            - Plans to harm themselves.
            - Recent suicide attempts.
            - Emergency situation related to suicide.
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
        if question_key in suicide_risk_questions:
            return True
        return False

def get_emergency_message():
    """Return emergency intervention message in standard format"""
    return {
        "message": "🚨 IMMEDIATE SAFETY CONCERN DETECTED 🚨. National Suicide Prevention Lifeline: 988 or 1-800-273-8255. Survey has been paused for your safety. Please seek immediate help."
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
        return user_response

async def extract_and_validate_multiple_answers(user_response: str, current_question_key: str, current_question_index: int) -> Dict[str, str]:
    """Extract multiple answers and validate each one individually"""
    try:
        future_questions = []
        for i, (question, key) in enumerate(questions):
            if i > current_question_index:
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
        
        validated_answers = {}
        
        for question_key, extracted_answer in extracted_answers.items():
            question_text = None
            for q_text, q_key in questions:
                if q_key == question_key:
                    question_text = q_text
                    break
            
            if question_text:
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
        return True

async def get_next_question(session):
    """Get the next question, handling pre-extracted answers"""
    current_index = session["question_index"]
    
    if current_index >= len(questions):
        return {
            "message": "Survey completed successfully!",
            "answers": session["answers"],
            "completed": True
        }
    
    question, key = questions[current_index]
    
    if key in session["extracted_answers"]:
        extracted_answer = session["extracted_answers"][key]
        session["confirmation_pending"] = key
        return {
            "message": f"I found that you previously mentioned: '{extracted_answer}' for the question '{question}'. Is this correct? Reply 'Yes' to confirm or provide a new answer."
        }
    
    return {"message": question}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Generate session ID and initialize session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "question_index": 0,
        "answers": {},
        "extracted_answers": {},
        "confirmation_pending": None,
        "greeting_done": False,
        "emergency_triggered": False,
        "websocket": websocket
    }
    
    # Send initial greeting
    await websocket.send_text(json.dumps({
        "message": "Hi! Let's start with some questions. Please respond with 'Hi' to begin."
    }))
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Try to parse as JSON, fallback to plain text
                try:
                    message_data = json.loads(data)
                    user_answer = message_data.get("message", data)
                except json.JSONDecodeError:
                    user_answer = data
                
                session = sessions[session_id]
                current_question_index = session["question_index"]
                
                # Check if emergency was already triggered
                if session.get("emergency_triggered", False):
                    await websocket.send_text(json.dumps(get_emergency_message()))
                    continue
                
                # Handle greeting
                if not session["greeting_done"]:
                    if user_answer.lower().strip() in ["hi", "hello", "start"]:
                        session["greeting_done"] = True
                        question, key = questions[0]
                        await websocket.send_text(json.dumps({"message": question}))
                        continue
                    else:
                        await websocket.send_text(json.dumps({
                            "message": "Please type 'Hi' to begin the survey."
                        }))
                        continue
                
                # General suicide risk check (after greeting is done)
                if session.get("greeting_done", False):
                    general_risk_check = await detect_suicide_risk(user_answer)
                    if general_risk_check:
                        session["emergency_triggered"] = True
                        await websocket.send_text(json.dumps(get_emergency_message()))
                        continue
                
                # Handle confirmation pending
                if session.get("confirmation_pending"):
                    confirmation_key = session["confirmation_pending"]
                    if user_answer.lower() in ["yes", "y", "correct", "that's right", "right"]:
                        # User confirmed the extracted answer
                        question, key = questions[current_question_index]
                        confirmed_answer = session["extracted_answers"][key]
                        
                        # Suicide risk check for confirmed answer
                        suicide_risk_detected = await detect_suicide_risk(confirmed_answer, key)
                        if suicide_risk_detected:
                            session["answers"][key] = confirmed_answer
                            session["emergency_triggered"] = True
                            await websocket.send_text(json.dumps(get_emergency_message()))
                            continue
                        
                        session["answers"][key] = session["extracted_answers"][key]
                        del session["extracted_answers"][confirmation_key]
                        session["confirmation_pending"] = None
                        session["question_index"] += 1
                        
                        # Move to next question
                        if session["question_index"] < len(questions):
                            response = await get_next_question(session)
                            await websocket.send_text(json.dumps(response))
                        else:
                            await websocket.send_text(json.dumps({
                                "message": "Survey completed successfully!",
                                "answers": session["answers"],
                                "completed": True
                            }))
                        continue
                    else:
                        # User wants to provide new answer
                        session["confirmation_pending"] = None
                        if confirmation_key in session["extracted_answers"]:
                            del session["extracted_answers"][confirmation_key]
                        
                        question, key = questions[current_question_index]
                        
                        # Validate the new answer
                        is_valid = await verify_answer(question, user_answer, expected_answers[key])
                        
                        if not is_valid:
                            await websocket.send_text(json.dumps({
                                "message": f"The answer for '{question}' seems inappropriate. Please provide a clearer answer."
                            }))
                            continue
                        
                        # Suicide risk check for new answer
                        suicide_risk_detected = await detect_suicide_risk(user_answer, key)
                        if suicide_risk_detected:
                            session["answers"][key] = user_answer
                            session["emergency_triggered"] = True
                            await websocket.send_text(json.dumps(get_emergency_message()))
                            continue
                        
                        # Save the new answer and continue
                        session["answers"][key] = user_answer
                        session["question_index"] += 1
                        
                        # Extract additional answers
                        validated_future_answers = await extract_and_validate_multiple_answers(
                            user_answer, key, current_question_index
                        )
                        if validated_future_answers:
                            # Check extracted answers for suicide risk
                            for future_key, future_answer in validated_future_answers.items():
                                suicide_risk_in_future = await detect_suicide_risk(future_answer, future_key)
                                if suicide_risk_in_future:
                                    session["emergency_triggered"] = True
                                    await websocket.send_text(json.dumps(get_emergency_message()))
                                    break
                            else:  # No suicide risk found in extracted answers
                                session["extracted_answers"].update(validated_future_answers)
                                print(f"✓ Extracted {len(validated_future_answers)} additional answers")
                        
                        if not session.get("emergency_triggered", False):
                            if session["question_index"] < len(questions):
                                response = await get_next_question(session)
                                await websocket.send_text(json.dumps(response))
                            else:
                                await websocket.send_text(json.dumps({
                                    "message": "Survey completed successfully!",
                                    "answers": session["answers"],
                                    "completed": True
                                }))
                        continue
                
                # Handle regular survey flow
                if current_question_index >= len(questions):
                    await websocket.send_text(json.dumps({
                        "message": "Survey completed successfully!",
                        "answers": session["answers"],
                        "completed": True
                    }))
                    continue
                
                # Get current question
                question, key = questions[current_question_index]
                
                # Check for pre-extracted answer
                if key in session["extracted_answers"]:
                    extracted_answer = session["extracted_answers"][key]
                    session["confirmation_pending"] = key
                    await websocket.send_text(json.dumps({
                        "message": f"I found that you previously mentioned: '{extracted_answer}' for the question '{question}'. Is this correct? Reply 'Yes' to confirm or provide a new answer."
                    }))
                    continue
                
                # Handle current answer - extract it from potentially multi-answer response
                current_answer = await validate_current_answer_from_multi_response(
                    user_answer, question, key
                )
                
                # Validate the current answer
                is_current_valid = await verify_answer(question, current_answer, expected_answers[key])
                
                if not is_current_valid:
                    await websocket.send_text(json.dumps({
                        "message": f"The answer for '{question}' seems inappropriate. Please provide a clearer answer for this specific question."
                    }))
                    continue
                
                # Suicide risk check for current answer
                suicide_risk_detected = await detect_suicide_risk(current_answer, key)
                if suicide_risk_detected:
                    session["answers"][key] = current_answer
                    session["emergency_triggered"] = True
                    await websocket.send_text(json.dumps(get_emergency_message()))
                    continue
                
                # Save current answer
                session["answers"][key] = current_answer
                
                # Extract and validate future answers
                validated_future_answers = await extract_and_validate_multiple_answers(
                    user_answer, key, current_question_index
                )
                
                if validated_future_answers:
                    # Check extracted answers for suicide risk
                    for future_key, future_answer in validated_future_answers.items():
                        suicide_risk_in_future = await detect_suicide_risk(future_answer, future_key)
                        if suicide_risk_in_future:
                            session["emergency_triggered"] = True
                            await websocket.send_text(json.dumps(get_emergency_message()))
                            break
                    else:  # No suicide risk found
                        session["extracted_answers"].update(validated_future_answers)
                        extracted_count = len(validated_future_answers)
                        print(f"✓ Successfully extracted and validated {extracted_count} future answers")
                
                if not session.get("emergency_triggered", False):
                    # Move to next question
                    session["question_index"] += 1
                    if session["question_index"] < len(questions):
                        response = await get_next_question(session)
                        await websocket.send_text(json.dumps(response))
                    else:
                        await websocket.send_text(json.dumps({
                            "message": "Survey completed successfully!",
                            "answers": session["answers"],
                            "completed": True
                        }))
                
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_text(json.dumps({
                    "message": "An error occurred processing your message. Please try again."
                }))
    
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        if session_id in sessions:
            del sessions[session_id]
    except Exception as e:
        print(f"WebSocket error: {e}")
        if session_id in sessions:
            del sessions[session_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

