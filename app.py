from datetime import date, datetime
from typing import Optional

import pymupdf
import streamlit as st
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EngineerOS",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONSTANTS
# =========================================================

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_NOTE_CHARACTERS = 45_000

STUDY_ACTIONS = [
    "Expand my notes",
    "Build a study guide",
    "Find missing concepts",
    "Create practice questions",
    "Explain with intuition",
    "Create flashcards",
]


# =========================================================
# SESSION STATE
# =========================================================

def initialize_session_state() -> None:
    """Create temporary storage used while the app is open."""

    defaults = {
        "notes_result": "",
        "tutor_result": "",
        "planner_result": "",
        "reflection_result": "",
        "interview_result": "",
        "task_list": [],
        "reflection_history": [],
        "interview_history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# =========================================================
# API KEY
# =========================================================

def get_groq_api_key() -> str:
    """
    Read the Groq API key from Streamlit Secrets.

    The key must be entered in Streamlit as:
    GROQ_API_KEY = "your-key"
    """

    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


GROQ_API_KEY = get_groq_api_key()


# =========================================================
# PDF FUNCTIONS
# =========================================================

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract machine-readable text from a PDF.

    This works best for typed notes and digital lecture slides.
    Scanned handwriting may not contain extractable text.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    extracted_pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            if page_text:
                extracted_pages.append(
                    f"\n\n--- PAGE {page_number} ---\n\n{page_text}"
                )
    finally:
        document.close()

    return "".join(extracted_pages).strip()


# =========================================================
# AI FUNCTIONS
# =========================================================

def call_ai(
    system_message: str,
    user_message: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 4_000,
) -> str:
    """
    Send a request to Groq and return the model's text response.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "Your Groq API key has not been added to "
            "Streamlit Secrets."
        )

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not response.choices:
        raise ValueError("The AI returned no response choices.")

    result = response.choices[0].message.content

    if not result:
        raise ValueError("The AI returned an empty response.")

    return result.strip()


def build_notes_instruction(selected_action: str) -> str:
    """Return instructions for the selected Notes Lab action."""

    instructions = {
        "Expand my notes": """
Expand incomplete explanations while preserving the student's
original structure.

Include:

1. What the notes already explain
2. Expanded explanations
3. Definitions that are missing
4. Equations and meanings of variables
5. Assumptions
6. Physical or molecular intuition
7. Engineering applications
8. Common mistakes
9. Questions the student should investigate

Clearly distinguish content found in the notes from information
you are adding.
""",
        "Build a study guide": """
Convert the notes into a structured study guide.

Include:

1. Learning objectives
2. Main concepts
3. Essential definitions
4. Important equations
5. Meaning and units of variables
6. Assumptions and limitations
7. Physical or molecular intuition
8. Conceptual examples
9. Common misconceptions
10. Final review checklist
""",
        "Find missing concepts": """
Identify ideas that are incomplete, unexplained, disconnected,
or likely to cause confusion.

For every missing area:

1. Identify the relevant topic
2. Explain what is missing
3. Supply a careful explanation
4. Explain why it matters
5. Give one active-recall question

Do not claim that added information appeared in the PDF.
""",
        "Create practice questions": """
Create an engineering practice set based on the notes.

Include:

1. Five conceptual questions
2. Five calculation or application questions when appropriate
3. Three intuition questions
4. Two misconception-detection questions
5. One challenging transfer problem

Place all solutions in a separate answer-key section at the end.
Do not make every question a simple definition question.
""",
        "Explain with intuition": """
Identify the most difficult concepts and explain each through:

1. Everyday intuition
2. Physical or molecular interpretation
3. Mathematical interpretation
4. Engineering application
5. Limiting cases
6. Important assumptions
7. Conditions under which the equation or model fails

Explain what happens when major variables increase or decrease.
""",
        "Create flashcards": """
Create useful active-recall flashcards.

Use this format:

Question:
Answer:

Include:

1. Definitions
2. Equation meanings
3. Assumptions
4. Conceptual relationships
5. Common mistakes
6. Physical intuition
7. Engineering applications

Create between 15 and 25 cards.
Avoid cards that are too long.
""",
    }

    return instructions[selected_action]


def analyze_notes(
    notes_text: str,
    selected_action: str,
    student_context: str,
    model_name: str,
) -> str:
    """Analyze extracted notes using the selected learning mode."""

    system_message = """
You are an engineering learning coach specializing in physics,
chemistry, mathematics, and chemical engineering.

Your purpose is to improve the student's reasoning, intuition,
retention, and technical communication.

Do not pretend that AI-generated material originally appeared in
the student's notes.

Do not fabricate sources, course requirements, experimental results,
or equations.

Use clear Markdown headings.

Technical equations and calculations should be presented carefully,
and the student should be reminded to verify high-stakes technical work.
"""

    action_instruction = build_notes_instruction(selected_action)

    user_message = f"""
STUDENT CONTEXT

{student_context or "No additional context was provided."}


SELECTED TASK

{selected_action}


TASK INSTRUCTIONS

{action_instruction}


STUDENT NOTES

{notes_text}
"""

    return call_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=model_name,
        temperature=0.25,
        max_tokens=5_000,
    )


def generate_tutor_response(
    subject: str,
    topic: str,
    student_attempt: str,
    tutor_mode: str,
    model_name: str,
) -> str:
    """Create a tutoring response for physics, chemistry, or engineering."""

    system_message = """
You are a Socratic engineering tutor.

Your purpose is to help students develop intuition and independent
reasoning rather than merely giving answers.

When a student provides an attempted solution:

1. Identify what is correct
2. Identify the first important reasoning error
3. Explain why it is an error
4. Give the smallest useful hint
5. Ask one question that moves the student forward

Do not provide a complete numerical solution unless the student
explicitly asks for a full solution or has already attempted the work.

Use equations when helpful, but explain their physical meaning.
"""

    mode_instructions = {
        "Socratic hints": """
Do not immediately solve the problem.
Give one useful hint and one question at a time.
""",
        "Intuition builder": """
Explain the concept through physical, molecular, mathematical,
and engineering perspectives.
""",
        "Equation explorer": """
Explain where the main equation comes from, what every variable
means, its assumptions, units, limiting cases, and failure conditions.
""",
        "Misconception check": """
Look for hidden misconceptions in the student's explanation.
Explain the misconception and provide a contrasting example.
""",
        "Full explanation": """
Give a complete teaching explanation, including reasoning,
equations, assumptions, and a final self-check question.
""",
    }

    user_message = f"""
SUBJECT

{subject}


TOPIC OR PROBLEM

{topic}


STUDENT'S CURRENT ATTEMPT OR UNDERSTANDING

{student_attempt or "The student has not provided an attempt yet."}


TUTOR MODE

{tutor_mode}


MODE INSTRUCTIONS

{mode_instructions[tutor_mode]}
"""

    return call_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=model_name,
        temperature=0.3,
        max_tokens=3_000,
    )


def create_schedule_analysis(
    tasks: list[dict],
    available_hours: float,
    energy_level: int,
    planning_notes: str,
    model_name: str,
) -> str:
    """Recommend a realistic daily study plan."""

    task_lines = []

    for index, task in enumerate(tasks, start=1):
        task_lines.append(
            f"""
Task {index}
Name: {task["name"]}
Subject: {task["subject"]}
Deadline: {task["deadline"]}
Estimated minutes: {task["minutes"]}
Difficulty: {task["difficulty"]}/5
Importance: {task["importance"]}/5
"""
        )

    task_text = "\n".join(task_lines)

    system_message = """
You are a workload analyst for an engineering student.

Build realistic plans rather than motivational or overly optimistic
schedules.

Prioritize deadlines, importance, difficulty, and cognitive energy.

Include breaks and transition time.

Do not schedule more work than can fit into the student's available time.

When the work cannot all fit, state what should be postponed or reduced.
"""

    user_message = f"""
AVAILABLE STUDY HOURS

{available_hours}


CURRENT ENERGY LEVEL

{energy_level}/10


TASKS

{task_text}


ADDITIONAL CONTEXT

{planning_notes or "No additional planning context was provided."}


CREATE:

1. Priority order
2. Recommended time blocks
3. Break schedule
4. What not to do today
5. Main risk to the plan
6. A simpler fallback plan
"""

    return call_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=model_name,
        temperature=0.2,
        max_tokens=2_500,
    )


def analyze_reflection(
    planned_work: str,
    completed_work: str,
    time_surprise: str,
    unclear_concept: str,
    distraction: str,
    energy_level: int,
    tomorrow_change: str,
    model_name: str,
) -> str:
    """Analyze a student's end-of-day reflection."""

    system_message = """
You are an engineering student's reflection coach.

Your job is to identify useful patterns without being judgmental.

Focus on:

1. Planning accuracy
2. Time-estimation errors
3. Energy patterns
4. Distractions
5. Knowledge gaps
6. One or two realistic improvements

Do not overwhelm the student with a long list of changes.
"""

    user_message = f"""
DATE

{date.today().isoformat()}


PLANNED WORK

{planned_work}


COMPLETED WORK

{completed_work}


WHAT TOOK LONGER OR SHORTER THAN EXPECTED

{time_surprise}


CONCEPT THAT REMAINS UNCLEAR

{unclear_concept}


MAIN DISTRACTION

{distraction}


ENERGY LEVEL

{energy_level}/10


STUDENT'S IDEA FOR TOMORROW

{tomorrow_change}


PROVIDE:

1. What went well
2. Main planning lesson
3. Main learning gap
4. One adjustment for tomorrow
5. One active-recall question
"""

    return call_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=model_name,
        temperature=0.25,
        max_tokens=2_000,
    )


def create_interview_feedback(
    interview_type: str,
    target_role: str,
    question: str,
    answer: str,
    model_name: str,
) -> str:
    """Evaluate an interview answer."""

    system_message = """
You are an engineering internship interview coach.

Give candid, constructive feedback.

For behavioral answers, evaluate:

- STAR structure
- Specific evidence
- Ownership
- Results
- Reflection
- Conciseness

For technical answers, evaluate:

- Problem framing
- Assumptions
- Technical accuracy
- Safety awareness
- Communication
- Ability to reason under uncertainty

Never invent experiences for the student.
Help improve how the student's real experience is communicated.
"""

    user_message = f"""
INTERVIEW TYPE

{interview_type}


TARGET ROLE OR COMPANY

{target_role or "General engineering internship"}


QUESTION

{question}


STUDENT ANSWER

{answer}


PROVIDE:

1. Overall score from 1 to 10
2. What was strong
3. What was unclear or weak
4. Missing evidence or technical reasoning
5. A stronger answer structure
6. A revised example using only facts already supplied
7. One follow-up question an interviewer may ask
"""

    return call_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=model_name,
        temperature=0.25,
        max_tokens=2_500,
    )


# =========================================================
# ERROR DISPLAY
# =========================================================

def display_ai_error(error: Exception) -> None:
    """Show a readable error without displaying a long traceback."""

    error_text = str(error)
    lower_error = error_text.lower()

    st.error("The AI request could not be completed.")

    if "api key" in lower_error or "authentication" in lower_error:
        st.warning(
            "Check that your Groq API key was copied correctly into "
            "Streamlit Secrets."
        )

    elif "rate limit" in lower_error or "429" in lower_error:
        st.warning(
            "The free API limit may have been reached. Wait briefly "
            "and try again with a smaller PDF."
        )

    elif "model" in lower_error and (
        "not found" in lower_error
        or "does not exist" in lower_error
    ):
        st.warning(
            "The selected model may no longer be available. "
            "Check Groq's supported-model list."
        )

    elif "too large" in lower_error or "context" in lower_error:
        st.warning(
            "The request may be too large. Try a shorter PDF or "
            "reduce the amount of extracted text."
        )

    else:
        st.warning(
            "Open Manage app → Logs if you need the detailed "
            "technical error."
        )

    with st.expander("Technical error details"):
        st.code(error_text)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🧪 EngineerOS")

    st.caption(
        "Your personal engineering learning system"
    )

    st.markdown("---")

    model_name = st.text_input(
        "AI model",
        value=DEFAULT_MODEL,
        help=(
            "This must be a currently supported Groq model ID."
        ),
    )

    if GROQ_API_KEY:
        st.success("Groq API key connected")
    else:
        st.error("Groq API key missing")

    st.markdown("---")

    st.subheader("Learning principle")

    st.write(
        """
Use AI to strengthen your thinking—not to replace it.

Attempt difficult problems before requesting complete solutions.
Verify important technical equations and calculations.
"""
    )

    st.markdown("---")

    st.caption(
        "Version 2.0 · Personal prototype"
    )


# =========================================================
# HEADER
# =========================================================

st.title("🧠 EngineerOS")

st.write(
    """
A personal AI workspace for engineering notes, intuition,
planning, reflection, and interview preparation.
"""
)

if not GROQ_API_KEY:
    st.warning(
        """
Before using the AI features, add your Groq key under:

**Manage app → Settings → Secrets**
"""
    )


# =========================================================
# MAIN NAVIGATION
# =========================================================

notes_tab, tutor_tab, planner_tab, reflection_tab, interview_tab = st.tabs(
    [
        "📚 Notes Lab",
        "🧠 Intuition Tutor",
        "📅 Daily Planner",
        "📝 Reflection",
        "💼 Interview Gym",
    ]
)


# =========================================================
# NOTES LAB
# =========================================================

with notes_tab:
    st.header("📚 Notes Lab")

    st.write(
        """
Upload typed notes or digital lecture slides and transform them
into an active learning resource.
"""
    )

    uploaded_file = st.file_uploader(
        "Upload one PDF",
        type=["pdf"],
        key="notes_pdf",
    )

    selected_action = st.selectbox(
        "Choose an action",
        STUDY_ACTIONS,
    )

    student_context = st.text_area(
        "Optional context",
        placeholder=(
            "Example: These are Thermodynamics notes about the "
            "first law. I understand the equation but struggle "
            "with sign conventions and physical intuition."
        ),
        height=110,
    )

    if uploaded_file is not None:
        try:
            extracted_text = extract_pdf_text(
                uploaded_file.getvalue()
            )

            if not extracted_text:
                st.error(
                    "No readable text was found. The PDF may be "
                    "scanned, handwritten, or image-based."
                )

            else:
                character_count = len(extracted_text)

                st.success(
                    f"PDF processed: approximately "
                    f"{character_count:,} characters extracted."
                )

                with st.expander("Preview extracted text"):
                    st.text_area(
                        "PDF text preview",
                        value=extracted_text[:15_000],
                        height=320,
                        disabled=True,
                    )

                if character_count > MAX_NOTE_CHARACTERS:
                    st.info(
                        f"This beginner version will analyze the first "
                        f"{MAX_NOTE_CHARACTERS:,} characters."
                    )

                notes_for_ai = extracted_text[
                    :MAX_NOTE_CHARACTERS
                ]

                if st.button(
                    "Analyze my notes",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(
                        "Building your study resource..."
                    ):
                        try:
                            st.session_state["notes_result"] = (
                                analyze_notes(
                                    notes_text=notes_for_ai,
                                    selected_action=selected_action,
                                    student_context=student_context,
                                    model_name=model_name,
                                )
                            )

                        except Exception as error:
                            display_ai_error(error)

        except Exception as error:
            st.error("The PDF could not be processed.")

            with st.expander("Technical error details"):
                st.code(str(error))

    if st.session_state["notes_result"]:
        st.markdown("---")
        st.subheader("Study Resource")

        st.markdown(
            st.session_state["notes_result"]
        )

        st.download_button(
            "Download study resource",
            data=st.session_state["notes_result"],
            file_name="engineering_study_resource.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =========================================================
# INTUITION TUTOR
# =========================================================

with tutor_tab:
    st.header("🧠 Intuition Tutor")

    st.write(
        """
Use this tutor to strengthen reasoning rather than immediately
receiving a final answer.
"""
    )

    tutor_col1, tutor_col2 = st.columns(2)

    with tutor_col1:
        subject = st.selectbox(
            "Subject",
            [
                "Physics",
                "General Chemistry",
                "Organic Chemistry",
                "Thermodynamics",
                "Fluid Mechanics",
                "Heat Transfer",
                "Mass Transfer",
                "Reaction Engineering",
                "Mathematics",
                "Other",
            ],
        )

    with tutor_col2:
        tutor_mode = st.selectbox(
            "Tutor mode",
            [
                "Socratic hints",
                "Intuition builder",
                "Equation explorer",
                "Misconception check",
                "Full explanation",
            ],
        )

    tutor_topic = st.text_area(
        "Concept, equation, or problem",
        placeholder=(
            "Example: Why does increasing temperature increase "
            "pressure in a rigid ideal-gas container?"
        ),
        height=140,
    )

    student_attempt = st.text_area(
        "Your attempt or current understanding",
        placeholder=(
            "Explain what you think is happening before asking "
            "the tutor for help."
        ),
        height=140,
    )

    if st.button(
        "Ask the tutor",
        type="primary",
        use_container_width=True,
    ):
        if not tutor_topic.strip():
            st.warning(
                "Enter a concept, equation, or problem first."
            )

        else:
            with st.spinner("Thinking through your problem..."):
                try:
                    st.session_state["tutor_result"] = (
                        generate_tutor_response(
                            subject=subject,
                            topic=tutor_topic,
                            student_attempt=student_attempt,
                            tutor_mode=tutor_mode,
                            model_name=model_name,
                        )
                    )

                except Exception as error:
                    display_ai_error(error)

    if st.session_state["tutor_result"]:
        st.markdown("---")
        st.subheader("Tutor Response")

        st.markdown(
            st.session_state["tutor_result"]
        )


# =========================================================
# DAILY PLANNER
# =========================================================

with planner_tab:
    st.header("📅 Daily Planner")

    st.write(
        """
Add your tasks, then generate a realistic study plan based on
available time, difficulty, importance, and energy.
"""
    )

    planner_col1, planner_col2 = st.columns(2)

    with planner_col1:
        available_hours = st.number_input(
            "Available study hours today",
            min_value=0.5,
            max_value=16.0,
            value=4.0,
            step=0.5,
        )

    with planner_col2:
        planning_energy = st.slider(
            "Current energy level",
            min_value=1,
            max_value=10,
            value=7,
        )

    st.subheader("Add a task")

    task_col1, task_col2 = st.columns(2)

    with task_col1:
        task_name = st.text_input(
            "Task name",
            placeholder="Complete Physics problem set",
        )

        task_subject = st.text_input(
            "Subject",
            placeholder="Physics",
        )

        task_deadline = st.date_input(
            "Deadline",
            value=date.today(),
        )

    with task_col2:
        task_minutes = st.number_input(
            "Estimated minutes",
            min_value=10,
            max_value=600,
            value=60,
            step=10,
        )

        task_difficulty = st.slider(
            "Difficulty",
            min_value=1,
            max_value=5,
            value=3,
        )

        task_importance = st.slider(
            "Importance",
            min_value=1,
            max_value=5,
            value=4,
        )

    if st.button("Add task"):
        if not task_name.strip():
            st.warning("Enter a task name.")

        else:
            st.session_state["task_list"].append(
                {
                    "name": task_name.strip(),
                    "subject": (
                        task_subject.strip()
                        or "Unspecified"
                    ),
                    "deadline": task_deadline.isoformat(),
                    "minutes": int(task_minutes),
                    "difficulty": int(task_difficulty),
                    "importance": int(task_importance),
                }
            )

            st.success("Task added.")

    if st.session_state["task_list"]:
        st.subheader("Today's task list")

        for task_index, task in enumerate(
            st.session_state["task_list"]
        ):
            task_text = (
                f"**{task['name']}** — "
                f"{task['subject']} · "
                f"{task['minutes']} min · "
                f"Difficulty {task['difficulty']}/5 · "
                f"Importance {task['importance']}/5 · "
                f"Due {task['deadline']}"
            )

            task_display_col, delete_col = st.columns(
                [9, 1]
            )

            with task_display_col:
                st.write(task_text)

            with delete_col:
                if st.button(
                    "✕",
                    key=f"delete_task_{task_index}",
                ):
                    st.session_state["task_list"].pop(
                        task_index
                    )
                    st.rerun()

        planning_notes = st.text_area(
            "Additional planning context",
            placeholder=(
                "Example: I have class until 2 PM and need a "
                "30-minute dinner break."
            ),
        )

        planner_button_col, clear_button_col = st.columns(2)

        with planner_button_col:
            if st.button(
                "Create my study plan",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner(
                    "Analyzing your workload..."
                ):
                    try:
                        st.session_state["planner_result"] = (
                            create_schedule_analysis(
                                tasks=st.session_state[
                                    "task_list"
                                ],
                                available_hours=available_hours,
                                energy_level=planning_energy,
                                planning_notes=planning_notes,
                                model_name=model_name,
                            )
                        )

                    except Exception as error:
                        display_ai_error(error)

        with clear_button_col:
            if st.button(
                "Clear all tasks",
                use_container_width=True,
            ):
                st.session_state["task_list"] = []
                st.session_state["planner_result"] = ""
                st.rerun()

    else:
        st.info(
            "Add at least one task to create a study plan."
        )

    if st.session_state["planner_result"]:
        st.markdown("---")
        st.subheader("Recommended Plan")

        st.markdown(
            st.session_state["planner_result"]
        )

        st.download_button(
            "Download today's plan",
            data=st.session_state["planner_result"],
            file_name=f"study_plan_{date.today()}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =========================================================
# REFLECTION
# =========================================================

with reflection_tab:
    st.header("📝 Daily Reflection")

    st.write(
        """
Compare what you planned with what actually happened.
Over time, this can help improve estimation and study decisions.
"""
    )

    planned_work = st.text_area(
        "What did you plan to complete?",
        height=100,
    )

    completed_work = st.text_area(
        "What did you actually complete?",
        height=100,
    )

    time_surprise = st.text_area(
        "What took longer or shorter than expected?",
        height=90,
    )

    unclear_concept = st.text_area(
        "What concept is still unclear?",
        height=90,
    )

    reflection_col1, reflection_col2 = st.columns(2)

    with reflection_col1:
        distraction = st.text_input(
            "Main distraction",
            placeholder="Phone, fatigue, unclear instructions...",
        )

    with reflection_col2:
        reflection_energy = st.slider(
            "Energy level today",
            min_value=1,
            max_value=10,
            value=6,
        )

    tomorrow_change = st.text_area(
        "What would you change tomorrow?",
        height=90,
    )

    if st.button(
        "Analyze my reflection",
        type="primary",
        use_container_width=True,
    ):
        if not planned_work.strip() and not completed_work.strip():
            st.warning(
                "Enter what you planned or completed first."
            )

        else:
            with st.spinner(
                "Finding useful patterns..."
            ):
                try:
                    result = analyze_reflection(
                        planned_work=planned_work,
                        completed_work=completed_work,
                        time_surprise=time_surprise,
                        unclear_concept=unclear_concept,
                        distraction=distraction,
                        energy_level=reflection_energy,
                        tomorrow_change=tomorrow_change,
                        model_name=model_name,
                    )

                    st.session_state["reflection_result"] = result

                    st.session_state[
                        "reflection_history"
                    ].append(
                        {
                            "timestamp": datetime.now().isoformat(
                                timespec="minutes"
                            ),
                            "reflection": result,
                        }
                    )

                except Exception as error:
                    display_ai_error(error)

    if st.session_state["reflection_result"]:
        st.markdown("---")
        st.subheader("Reflection Analysis")

        st.markdown(
            st.session_state["reflection_result"]
        )

        reflection_download = f"""
# Daily Reflection

Date: {date.today().isoformat()}

## Planned Work

{planned_work}

## Completed Work

{completed_work}

## Coach Analysis

{st.session_state["reflection_result"]}
"""

        st.download_button(
            "Download reflection",
            data=reflection_download,
            file_name=f"reflection_{date.today()}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =========================================================
# INTERVIEW GYM
# =========================================================

with interview_tab:
    st.header("💼 Interview Gym")

    st.write(
        """
Practice behavioral or technical engineering questions and
receive structured feedback.
"""
    )

    interview_col1, interview_col2 = st.columns(2)

    with interview_col1:
        interview_type = st.selectbox(
            "Interview type",
            [
                "Behavioral",
                "Technical",
                "Project explanation",
                "Safety scenario",
                "Troubleshooting scenario",
            ],
        )

    with interview_col2:
        target_role = st.text_input(
            "Target role or company",
            placeholder=(
                "Process Engineering Intern at Dow"
            ),
        )

    default_questions = {
        "Behavioral": (
            "Tell me about a time you faced a difficult "
            "team challenge."
        ),
        "Technical": (
            "How would you approach sizing a heat exchanger?"
        ),
        "Project explanation": (
            "Tell me about a technical project you completed."
        ),
        "Safety scenario": (
            "What would you do if you observed an unsafe "
            "condition in a chemical plant?"
        ),
        "Troubleshooting scenario": (
            "A process flow rate suddenly decreases. "
            "How would you investigate the problem?"
        ),
    }

    interview_question = st.text_area(
        "Interview question",
        value=default_questions[interview_type],
        height=100,
    )

    interview_answer = st.text_area(
        "Your answer",
        placeholder=(
            "Write your answer as if you were speaking "
            "to the interviewer."
        ),
        height=220,
    )

    if st.button(
        "Evaluate my answer",
        type="primary",
        use_container_width=True,
    ):
        if not interview_answer.strip():
            st.warning(
                "Write your interview answer first."
            )

        else:
            with st.spinner(
                "Evaluating your response..."
            ):
                try:
                    result = create_interview_feedback(
                        interview_type=interview_type,
                        target_role=target_role,
                        question=interview_question,
                        answer=interview_answer,
                        model_name=model_name,
                    )

                    st.session_state["interview_result"] = result

                    st.session_state[
                        "interview_history"
                    ].append(
                        {
                            "timestamp": datetime.now().isoformat(
                                timespec="minutes"
                            ),
                            "type": interview_type,
                            "question": interview_question,
                            "feedback": result,
                        }
                    )

                except Exception as error:
                    display_ai_error(error)

    if st.session_state["interview_result"]:
        st.markdown("---")
        st.subheader("Interview Feedback")

        st.markdown(
            st.session_state["interview_result"]
        )

        interview_download = f"""
# Interview Practice

## Type

{interview_type}

## Target Role

{target_role or "General engineering internship"}

## Question

{interview_question}

## My Answer

{interview_answer}

## Feedback

{st.session_state["interview_result"]}
"""

        st.download_button(
            "Download interview feedback",
            data=interview_download,
            file_name="interview_feedback.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    """
EngineerOS is a learning assistant, not an authoritative engineering
reference. Verify technical equations, calculations, safety decisions,
and course requirements using trusted sources.
"""
)
