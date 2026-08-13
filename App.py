#App.py

from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
import re
import tempfile
import logging
import subprocess
import wave
import math
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from together import Together


# --------------------------------------------------
# Configuration
# --------------------------------------------------


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

#Together AI client
client = Together(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    timeout=300
)


# --------------------------------------------------
# Routes
# --------------------------------------------------


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'healthy'}


# --------------------------------------------------
# Audio processing
# --------------------------------------------------


def split_audio_into_chunks(audio_file_path, chunk_duration_seconds=25):
    """
    Split audio file into smaller chunks for processing.
    """
    try:
        with wave.open(audio_file_path, 'rb') as wav_file:

            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            total_duration = frames / sample_rate
            chunk_frames = int(chunk_duration_seconds * sample_rate)

            number_of_chunks = math.ceil(total_duration/chunk_duration_seconds) 
            
            logger.info(
                f"Audio duration: {total_duration:.2f}s, "
                f"will create {number_of_chunks} chunks"
            )
            
            chunk_files = []
            
            for i in range(0, frames, chunk_frames):
                
                chunk_number = i // chunk_frames

                chunk_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=f'_chunk_{chunk_number}.wav'
                )

                chunk_path = chunk_file.name
                chunk_file.close()
                
                wav_file.setpos(i)

                chunk_data = wav_file.readframes(
                    min(chunk_frames, frames - i)
                )
                
                with wave.open(chunk_path, 'wb') as chunk_wav:
                    chunk_wav.setnchannels(channels)
                    chunk_wav.setsampwidth(sample_width)
                    chunk_wav.setframerate(sample_rate)
                    chunk_wav.writeframes(chunk_data)
                
                chunk_files.append(chunk_path)

                logger.info(
                    f"Created chunk {len(chunk_files)}: {chunk_path}"
                )
        
        return chunk_files
        
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        return []

def process_audio_chunk(chunk_path, chunk_index):
    """
    Transcribe one audio chunk using Google Speech Recognition
    """
    try:
        logger.info(
            f"Processing chunk {chunk_index}: {chunk_path}"
        )
        
        recognizer = sr.Recognizer()

        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = False  
        recognizer.pause_threshold = 0.5
        recognizer.operation_timeout = 30  
        
        with sr.AudioFile(chunk_path) as source:
            
            recognizer.adjust_for_ambient_noise(
                source, 
                duration=0.5
            )

            audio_data = recognizer.record(source)
            
            
            try:

                text = recognizer.recognize_google(
                    audio_data, 
                    language='en-US'
                )

                logger.info(
                    f"Chunk {chunk_index} processed successfully: "
                    f"{len(text)} chars"
                )

                return True, text, None
            
            except sr.UnknownValueError:

                logger.warning(
                    f"Chunk {chunk_index}: No speech detected"
                )

                return True, "", None  
            
            except sr.RequestError as e:

                logger.error(
                    f"Chunk {chunk_index}: Service error: {e}"
                )
                return False, "", str(e)
                
    except Exception as e:
        logger.error(
            f"Error processing chunk {chunk_index}: {e}"
        )

        return False, "", str(e)


def process_long_audio(audio_file_path, max_duration=900):
    """
    Transcribe audio.
    Audio longer than 25 seconds is split into chunks.
    Chunks are processed in batchets of two.
    """
    try:

        with wave.open(audio_file_path, 'rb') as wav_file:

            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()

            duration = frames / sample_rate
            
            logger.info(
                f"Audio duration: {duration:.2f} seconds"
            )
            #Limit recordings to 15 minutes
            if duration > max_duration:

                logger.warning(
                    f"Audio too long {max_duration}s. "
                    f"Truncating."
                )
                
                truncated_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix='_truncated.wav'
                )

                truncated_path = truncated_file.name
                truncated_file.close()
                
                truncated_frames = int(
                    max_duration * sample_rate
                )

                truncated_data = wav_file.readframes(
                    truncated_frames
                )
                
                with wave.open(
                    truncated_path, 
                    'wb'
                ) as truncated_wav:
                    
                    truncated_wav.setnchannels(
                        wav_file.getnchannels()
                    )

                    truncated_wav.setsampwidth(
                        wav_file.getsampwidth()
                    )

                    truncated_wav.setframerate(
                        sample_rate
                    )

                    truncated_wav.writeframes(
                        truncated_data
                    )
                
                audio_file_path = truncated_path
                duration = max_duration
        
        # Short recordings
        if duration <= 25:

            logger.info(
                "Short audio, processing directly"
            )

            return process_audio_chunk(
                audio_file_path, 
                0
            )
        
        # Long recordings
        logger.info(
            "Long audio detected, splitting into chunks"
        )

        chunk_files = split_audio_into_chunks(
            audio_file_path, 
            chunk_duration_seconds=25
        )
        
        if not chunk_files:

            return (
                False, 
                "", 
                "Failed to split audio into chunks"
            )
        
        transcripts = []
        errors = []
        
        chunk_args = [
            (chunk_path, i) 
            for i, chunk_path 
            in enumerate(chunk_files)
        ]
        
        # Process two batched at a time
        batch_size = 2

        for i in range(
            0, 
            len(chunk_args), 
            batch_size
        ):
            
            batch = chunk_args[
                i:i + batch_size
            ]
            
            with ThreadPoolExecutor(
                max_workers=batch_size
            ) as executor:
                
                results = list(executor.map(
                    lambda args: process_audio_chunk(
                        args[0],
                        args[1]
                    ), 
                    batch
                )
            )
            
            for success, transcript, error in results:

                if success:

                    if transcript:  
                        transcripts.append(transcript)

                else:
                    errors.append(error)
            
            # Small delay between batches
            time.sleep(0.5)
        
        # Clean up chunk files
        for chunk_path in chunk_files:

            try:
                os.unlink(chunk_path)
            except OSError:
                pass
        
        # No successful transcription
        if not transcripts: 

            if errors:
                return (
                    False, 
                    "", 
                    f"All chunks failed: "
                    f"{'; '.join(errors[:3])}"
                )
            
            return (
                False, 
                "", 
                "No speech detected"
            )
        
        # Combine transcripts
        combined_transcript = ' '.join(
            transcripts
        )

        logger.info(
            f"Combined transcript: "
            f"{len(combined_transcript)} characters "
            f"from {len(transcripts)} chunks"
        )
        
        if len(combined_transcript.strip()) < 10:

            return (
                False, 
                "", 
                "Combined transcript too short"
            )
        
        return (
            True, 
            combined_transcript, 
            None
        )
        
    except Exception as e:

        logger.error(
            f"Error in parallel processing: {e}"
        )

        return False, "", str(e)
    

# --------------------------------------------------
# Audio validation and conversion
# --------------------------------------------------


def validate_audio_file(file_path):
    """
    Validate that the converted WAV file is usable
    """
    try:

        file_size = os.path.getsize(file_path)

        if file_size == 0:

            logger.error(
                "Audio file is empty"
            )

            return False
        
        with wave.open(
            file_path, 
            'rb'
        ) as wav_file:
            
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()

            channels = wav_file.getnchannels()
            
            duration = (
                frames / frame_rate 
                if frame_rate > 0 
                else 0
            )
            
            logger.info(
                f"Audio details: "
                f"{frames} frames, "
                f"{duration:.2f}s duration, "
                f"{frame_rate}Hz, "
                f"{channels} channels"
            )
            
            if frames == 0:

                logger.error(
                    "Audio file contains no frames"
                )

                return False
            
            if duration < 0.1:  

                logger.warning(
                    "Audio duration very short"
                )

                return False
            
            return True
        
    except Exception as e:
        logger.error(
            f"Audio validation error: {e}"
        )

        return False

def convert_to_wav_ffmpeg(
        input_path, 
        output_path
    ):
    """ 
    Use system ffmpeg 
    """
    
    try:
        logger.info(
            f"Converting audio using ffmpeg"
        )

        subprocess.run([
            "ffmpeg",
            '-y',
            '-i', 
            input_path,
            '-ar', 
            '16000',
            '-ac', 
            '1',
            '-f', 
            'wav',
            output_path
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        check=True, 
        timeout=30
        )

        logger.info(
            "ffmpeg conversion successful"
        )

        return True

    except subprocess.CalledProcessError as e:

        logger.error(
            f"ffmpeg conversion failed: "
            f"{e.stderr.decode('utf-8')}"
        )

        return False
    
    except FileNotFoundError as e:

        logger.error(
            "ffmpeg was not found"
        )

        return False
    
    except subprocess.TimeoutExpired:

        logger.error(
            "ffmpeg conversion timed out"
        )

        return False


# --------------------------------------------------
# Main audio processing
# --------------------------------------------------


@app.route(
        "/process_audio", 
        methods=['POST']
)
def process_audio():

    """
    Receive audio, transcribe it.
    Generate an HPI, and return both
    """

    logger.info(
        "Processing audio request received"
    )

    temp_file_path = None
    converted_path = None

    try:

        # ------------------------------------------
        # Validate upload
        # ------------------------------------------

        if "audio" not in request.files:
            
            return jsonify({
                "error": "No audio file provided"
            }), 400

        audio_file = request.files["audio"]

        if audio_file.filename == "":

            return jsonify({
                "error": "No audio file selected"
            }), 400
        
        # ------------------------------------------
        # Save uploaded audio
        # ------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix='.webm'
        ) as temp_file:
            
            audio_file.save(
                temp_file.name
            )
            
            temp_file_path = temp_file.name

            logger.info(
                f"Original audio file saved: "
                f"{temp_file_path}"
            )

        # ------------------------------------------
        # Convert to WAV
        # ------------------------------------------
        
        converted_path = (
            temp_file_path
            .replace('.webm', '.wav')
        )

        if not convert_to_wav_ffmpeg(
            temp_file_path, 
            converted_path
        ):
            
            return jsonify({
                "error": 
                "Failed to convert audio"
            }), 500

        if not validate_audio_file(
            converted_path
        ):
            
            return jsonify({
                "error": 
                "Invalid or unsupported audio format"
            }), 400

        # ------------------------------------------
        # Transcribe
        # ------------------------------------------
        
        success, transcript, error_message = (
            process_long_audio(
                converted_path
            )
        )

        if not success:

            logger.error(
                f"Speech recognition failed: "
                f"{error_message}"
            )

            return jsonify({
                "error": error_message
            }), 400
        
        logger.info(
            "Audio transcription completed"
        )

        # ------------------------------------------
        # Generate HPI
        # ------------------------------------------

        hpi = generate_hpi(
            transcript
        )

        # ------------------------------------------
        # Return result
        # ------------------------------------------

        return jsonify({
            #The frontend can choose wheter to display it
            "transcript": transcript,

            #Main result displayed to doctor
            "hpi": hpi
        })

    except Exception as e:

        import traceback
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())   # <-- add this line
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    finally:
        # ------------------------------------------
        # Clean temporary files
        # ------------------------------------------
        for file_path in [
            temp_file_path,
            converted_path
        ]:
            if file_path and os.path.exists(
                file_path
            ):
                try:

                    os.unlink(file_path)

                except OSError as e:
                    logger.warning(
                        f"Could not delete "
                        f"{file_path}: {e}"
                    )
    

# --------------------------------------------------
# AI HPI generation
# --------------------------------------------------


def clean_ai_response(text):
    """
    Clean unnecessary AI commentary.
    """

    if not text:
        return ""
    
    #Remove Qwen thing blocks
    text = re.sub(
        r'<think>.*?</think>', 
        "",
        text, 
        flags=re.DOTALL | re.IGNORECASE
    )

    #Remove markdown formatting
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text
    )
    
    text = text.strip()

    # If the model added text before HPI,
    # keep only the HPI section.
    hpi_match = re.search(
        r"HISTORY OF PRESENT ILLNESS\s*:?",
        text,
        flags=re.IGNORECASE
    )

    if hpi_match:

        text = text[
            hpi_match.start():
        ]

        # Remove the heading so we can
        # provide a clean result.
        text = re.sub(
            r"^HISTORY OF PRESENT ILLNESS\s*:?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

    # Remove accidental physical exam section
    text = re.split(
        r"PHYSICAL EXAMINATION\s*:?",
        text,
        flags=re.IGNORECASE
    )[0]
    
    return text.strip()

def generate_hpi(transcript):
    """
    Generate a clinical History of Present Illnes from transcript.
    """

    prompt = f"""
    You are an experienced emergency department physician creating clinical documentation from a patient interview.

    Write ONLY a professional History of Present Illness (HPI) based on the transcript below.

    PATIENT REFERENCE:
    - Always refer to the patient as "the patient" or use they/them pronouns.
    - Do not assume or document the patient's gender identity.

    GENERAL HPI REQUIREMENTS:

    Include relevant information from the transcript regarding:
    - Chief complaint
    - Onset and duration
    - Location
    - Quality and severity
    - Associated symptoms
    - Pertinent positive findings
    - Pertinent negative findings
    - Relevant medical history
    - Relevant surgical history
    - Relevant medications
    - Allergies
    - Relevant social history
    - Relevant family history

    Only document details that are actually present in the transcript.

    If a general HPI detail such as location, severity, onset, duration, or associated symptoms is not mentioned in the transcript, DO NOT state that it was not mentioned, unknown, unspecified, or unavailable. Simply omit that information.

    Do not add unnecessary filler or statements that do not provide clinically useful information.

    RED FLAG REQUIREMENTS:

    The purpose of documenting red flags is to identify symptoms or risk factors relevant to potentially serious or life-threatening conditions.

    For the chief complaint identified in the transcript, document the relevant red flags listed below.

    IMPORTANT RED FLAG RULE:
    - If a red flag is explicitly positive in the transcript, document it as positive.
    - If a red flag is explicitly negative in the transcript, document it as negative.
    - If a red flag is NOT mentioned anywhere in the transcript, document it as negative when it belongs to the relevant red-flag list for the patient's chief complaint.
    - Do not invent positive symptoms or risk factors.
    - Do not document a red flag as both positive and negative.
    - Never create contradictory statements. If the transcript indicates a positive finding, do not subsequently document the same finding as negative.

    CHIEF COMPLAINT-SPECIFIC RED FLAGS:

    1. ABDOMINAL PAIN

    Relevant red flags include:
    - Vomiting
    - Fever
    - Bowel or bladder pattern changes
    - GI or GU bleeding, including vaginal bleeding, rectal bleeding, or hematuria
    - Unexplained weight loss
    - Sick contacts
    - Recent travel
    - Excessive drug, smoking, or alcohol use

    2. CHEST PAIN OR SHORTNESS OF BREATH

    Relevant red flags include:
    - Lightheadedness
    - Syncope
    - Neurological symptoms
    - Chest pain
    - Dyspnea
    - Fever
    - Cough
    - Vomiting
    - Significant fatigue
    - Excessive drug, smoking, or alcohol use
    - Sick contacts
    - Recent travel
    - Tearing sensation to the back
    - Pulmonary embolism risk factors

    For chest pain or shortness of breath, explicitly document:
    "No pulmonary embolism risk factors"
    when no pulmonary embolism risk factors are discussed in the transcript.

    Also explicitly document:
    "No first-degree cardiac event at age <55M/<65F"
    when this family history is not discussed in the transcript.

    If the transcript does mention a pulmonary embolism risk factor or a first-degree cardiac event at age <55M/<65F, document the positive or relevant finding instead and do not state the corresponding negative.

    3. HEADACHE OR NEUROLOGICAL COMPLAINT

    Relevant red flags include:
    - Thunderclap headache
    - Fever
    - Neck pain or stiffness
    - Vomiting
    - Seizure
    - Syncope
    - Vision changes
    - Facial weakness
    - Extremity weakness
    - Facial or extremity numbness
    - Ataxia
    - Excessive drug, smoking, or alcohol use

    4. BACK PAIN

    Relevant red flags include:
    - Fever
    - Limb weakness
    - Inability to ambulate
    - Saddle anesthesia
    - Urinary incontinence
    - Fecal incontinence
    - Urinary retention
    - Fecal retention
    - Excessive drug, smoking, or alcohol use

    5. RESPIRATORY OR FLU-LIKE PRESENTING COMPLAINT

    Relevant red flags include:
    - Shortness of breath
    - Chest pain
    - Purulent discharge
    - Neck stiffness
    - Fever lasting more than 5 days
    - Lethargy
    - Vomiting
    - Excessive drug, smoking, or alcohol use
    - Sick contacts
    - Recent travel

    6. OTHER CHIEF COMPLAINTS

    For other presenting complaints, identify the chief complaint and document the most clinically relevant red flags associated with potentially serious or life-threatening conditions.

    Only include red flags that are relevant to the presenting complaint. Do not add an unnecessarily long list of unrelated negative symptoms.

    CONTRADICTION PREVENTION:

    Before producing the final HPI, check the transcript for contradictions.

    For every symptom or red flag:
    - If it is documented as positive, do not document it as negative.
    - If it is explicitly denied, document it as negative when clinically relevant.
    - Do not infer a positive finding that is not supported by the transcript.
    - Do not create conflicting statements.

    DIAGNOSIS:

    Do NOT make a diagnosis or differential diagnosis unless a diagnosis has already been explicitly provided by a clinician in the transcript.

    Do not interpret symptoms as a diagnosis.

    Do NOT create:
    - Physical Examination
    - Assessment
    - Plan
    - Medical advice
    - Recommendations
    - Commentary
    - Explanation of reasoning
    - AI reasoning or analysis

    OUTPUT FORMAT:

    Return ONLY the HPI.

    Begin directly with:

    HISTORY OF PRESENT ILLNESS:

    Write the HPI in concise, professional emergency-department documentation style.

    Do not include headings for Physical Examination, Assessment, Plan, or any other section.

    TRANSCRIPT:
    {transcript}
    """

    try:
        response = client.chat.completions.create(

            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",  

            messages=[
                {
                    "role": "system",
                    "content": """You are an experienced emergency department physician.
                                Generate ONLY a professional History of Present Illness (HPI).
                                Never generate a physical examination, assessment, plan, diagnosis,
                                medical advice, or commentary unless explicitly provided as a diagnosis
                                by a clinician in the transcript.
                                Always refer to the patient as "the patient" or use they/them pronouns."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            stream=False,

            temperature=0.2, 

            max_tokens=1000
        )
        
        generated_text = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        logger.info(f"Raw AI response: {generated_text[:500]}")

        cleaned_hpi = clean_ai_response(
            generated_text
        )

        logger.info(f"Cleaned HPI: {cleaned_hpi[:500]}")
        
        logger.info(
            "HPI generated successfully"
        )

        return cleaned_hpi
            
    except Exception as e:
        logger.error(
            f"Error generating HPI: {e}"
        )

        return (
            "Unable to generate the HPI automatically. "
            "Please review the transcript or API connection."
        )



# --------------------------------------------------
# Local development
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8080
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )