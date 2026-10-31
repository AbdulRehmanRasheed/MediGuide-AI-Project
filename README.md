🩺 MediGuide AI
AI-powered symptom assessment and patient guidance assistant

Safety-first, multilingual health guidance built with LangChain, Google Gemini and Streamlit.

Show Image Show Image Show Image Show Image Show Image

</div>
⚠️ Medical disclaimer

MediGuide AI is an educational prototype. It is not a doctor, not a medical device, and it never gives a confirmed diagnosis. It must not be used for real diagnosis or treatment. Always consult a qualified healthcare professional. In an emergency, call your local emergency number immediately (for example 911, 112 or 1122).

📖 Overview

MediGuide AI lets a user describe their symptoms in a simple form and returns preliminary, educational guidance: an urgency level, possible conditions (for learning only), recommended next steps, questions to ask a doctor, and warning signs to watch for.

The model's judgement is never trusted blindly. A rule-based safety layer can raise the urgency level (never lower it) when it detects red-flag symptoms such as chest pain with shortness of breath, and it tells the user why.

The project is a programming assignment that demonstrates the core LangChain building blocks in one readable app: PromptTemplate, ChatPromptTemplate, message types, LLMChain, streaming, and LLM caching.

✨ Features
Area	What it does
Structured assessment	The model returns strict JSON that is parsed and validated safely. Bad or missing output never crashes the app.
Urgency levels	LOW, MEDIUM, HIGH or EMERGENCY, shown with colour, clear actions and an alert style that matches the level.
Safety floor	Built-in red-flag rules (high-risk symptoms, severity 9 or more, emergency phrases in free text) raise the urgency and explain why. Simple negations such as "no chest pain" are handled.
Live narrative	A plain-language explanation is streamed token by token into the page.
Follow-up chat	Ask follow-up questions. The earlier result is sent back as an AIMessage, so the model remembers what it said.
Multilingual	English, Urdu, Hindi, Arabic, Spanish, French, German, Turkish and Chinese (Simplified). Urdu and Arabic render right to left.
Caching	Choose Off, In-memory or SQLite (disk). Cache hits are shown in the results, and caches can be cleared from the sidebar.
Model choice	Pick a Gemini model and temperature in the sidebar, or type a custom model name.
Transparency	An "Under the hood" panel shows the exact messages sent to the model.
Report download	Export the full result as a .txt report that always includes the disclaimer.
Session history	Keeps a short history of recent checks in the sidebar.
Friendly errors	Invalid keys, quota limits, missing models, blocked requests and connection problems are explained in plain language.
Clinical light theme	A custom blue and teal interface with an ECG-style hero, tuned to stay readable in any browser theme.
🧱 Tech stack
UI: Streamlit
LLM orchestration: LangChain (langchain, langchain-core, langchain-community)
Model provider: Google Gemini through langchain-google-genai (ChatGoogleGenerativeAI)
Caching: LangChain InMemoryCache and SQLiteCache
Config: python-dotenv for the API key
🔄 How it works
form input
   │
   ▼
build patient profile  ──►  PromptTemplate
   │
   ▼
select cache (Off / In-memory / SQLite)
   │
   ▼
ChatPromptTemplate (System + Human messages)  ──►  LLMChain  ──►  Gemini
   │
   ▼
parse + normalise JSON  ──►  apply safety floor (never lowers urgency)
   │
   ▼
stream narrative  ──►  render dashboard  ──►  follow-up chat (AIMessage memory)

The code lives in a single file, app.py, split into clearly marked sections that map one-to-one to the modules suggested by the assignment, so it is easy to split into a src/ package later:

Section in app.py	Suggested module	Purpose
1. Config	src/config.py	Settings, form options, urgency styles
2. Utils	src/utils.py	Safe JSON parsing, safety rules, helpers
3. Prompts	src/prompts.py	PromptTemplate and ChatPromptTemplate
4. Chains	src/chains.py	LLM, LLMChain, streaming, message demo
5. Cache manager	src/cache_manager.py	In-memory and SQLite caches
6. Streamlit UI	app.py	Sidebar, form and results dashboard
🚀 Getting started
1. Prerequisites
Python 3.10 or newer
A Google Gemini API key. Create one for free at Google AI Studio.
2. Clone and install
bash
git clone https://github.com/<your-username>/MediGuide-AI-Project.git
cd MediGuide-AI-Project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
3. Add your API key

Copy the example file and put your key in it:

bash
cp .env.example .env
env
GOOGLE_API_KEY=your_api_key_here

Never commit your .env file. It is already listed in .gitignore. If no key is found in .env, the sidebar shows a password field where you can paste one for the session.

4. Run
bash
streamlit run app.py

Open the address Streamlit prints, usually http://localhost:8501.

Theme note

The repository includes .streamlit/config.toml, which locks Streamlit to the light theme so the form controls render correctly. Streamlit reads this file from the folder where you run the command, so run streamlit run app.py from the project root.

🗂️ Project structure
MediGuide-AI-Project/
├── app.py                  # The whole application (config, utils, prompts, chains, cache, UI)
├── requirements.txt        # Python dependencies
├── .env.example            # Template for your API key
├── .gitignore
├── .streamlit/
│   └── config.toml         # Light theme settings
└── README.md

A mediguide_cache.db file is created next to app.py when you use the SQLite cache. It is ignored by Git.

🧪 Manual test scenarios
Scenario	Expected result
Click the button with no symptoms selected	A warning appears and no API call is made
Enter an invalid age such as abc or 150	A validation message appears
Select Chest pain and Shortness of breath	Urgency is raised to EMERGENCY and the safety-rule banner explains why
Type "no chest pain" in the notes	The negation is recognised and does not trigger the high-risk rule
Set severity to 9 or 10	Urgency is at least HIGH
Run the same input twice with a cache enabled	The second run is faster and shows a "cache hit"
Choose Urdu or Arabic as the answer language	The results switch to right-to-left layout
Ask a follow-up question	The reply uses the earlier assessment as context
Remove the API key	A clear message asks for a key instead of crashing
🛡️ Safety design
The prompts forbid confirmed diagnoses, prescription drug names and doses, and tell the model to treat patient text as data, not instructions (a basic defence against prompt injection).
When unsure between two urgency levels, the model is told to choose the higher one.
The rule-based safety floor can only raise the urgency, never lower it.
If the model returns no clear urgency, the app falls back to a cautious MEDIUM.
The disclaimer appears in the sidebar, the page header, the results and the downloaded report.
⚠️ Limitations
Output quality depends on the Gemini model and can be wrong or incomplete.
The safety rules are a simple keyword list, not a clinical triage system.
Gemini's own safety filters may occasionally block a medical request. If that happens, reword the notes and try again.
Not validated clinically and not intended for real patients.
🛣️ Possible improvements
Split app.py into the src/ package.
Add unit tests for parse_assessment, apply_safety_floor and validate_age.
Use structured output (with_structured_output) instead of prompt-based JSON.
Add a dark theme option.
Add PDF export for the report.
🤝 Contributing

This is an educational project. Issues and pull requests that improve safety, clarity or accessibility are welcome.

📄 License

Released under the MIT License. Add a LICENSE file to the repository root before publishing.
