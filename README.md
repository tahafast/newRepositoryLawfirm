# Create a virtual environment
python -m venv myenv

# Activate the virtual environment
# On macOS/Linux
source myenv/bin/activate
# On Windows
.\myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload


Law Firm Chatbot/
├── app/
│   ├── common/
│   │   └── response/
│   ├── modules/
│   │   ├── brand_culture_strategy/
│   │   │   └── api/
│   │   └── router.py
│   └── services/
│       └── apollo/
├── core/
│   └── conf.py
│   └── logging_conf.py
├── database/
│   └── mongodb.py
├── main.py
├── requirements.txt
└── README.md