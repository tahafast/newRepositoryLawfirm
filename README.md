# Create a virtual environment
python -m venv robi

# Activate the virtual environment
# On macOS/Linux
source robi/bin/activate
# On Windows
.\robi\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload


Robi_migration/
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
├── database/
│   └── mongodb.py
├── main.py
├── requirements.txt
└── README.md