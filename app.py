import streamlit as st
import pandas as pd
import gspread
import matplotlib.pyplot as plt
from google.oauth2.service_account import Credentials
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. HARDCODED CONFIGURATION
# ==========================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GOOGLE_SHEET_URL = st.secrets["GOOGLE_SHEET_URL"]
#SERVICE_ACCOUNT_FILE = "C:\Users\Razer55\Downloads\Workout_LLM\API_Keys\credentials.json"
APP_PASSWORD = st.secrets["APP_PASSWORD"]

# --- SYSTEM PROMPT ---
SYSTEM_CONTEXT = """
You are a senior data analyst. 
1. When asked for trends, averages, or comparisons, only create a chart when the user specifies that is what they wish to see (line chart, bar chart, or histogram) using matplotlib.
2. Use clear labels and titles for any charts you create.
3. If you create a chart, finish your thought by explaining what the chart shows.
4. If a calculation is requested, show the steps.
5. If you are unsure what the user is asking, always ask clarifying questions until you are sure you can answer accurately (Ex. what exercise the user is asking about or what time frame)
6. Understand that the questions will be based on a workout log data set, so use your best knowledge to use and understand the terms popular in the world of strength training and bodybuilding
7. Only answer questions related to this dataset

Use the following data dictionary to understand the data:
-Row meaning: each unique row represents a distinct set that was performed
- Column A - "Date" (date): The date that the set was performed on
- Column B - "Set_ID" (string): The unique identifier for each set
- Column C - "Gym_Day" (string): The muscles that were targetted that day in the gym in which the set was performed, lower and legs despite having different names can be treated as the same gym day (based on the Push, Pull, Legs, Rest, Upper, Lower, Rest workout split)
- Column D - "Exersize" (string): The name of the exersize that was performed
- Column E - "Type" (string): If the intended outcome of the exersize was either Hypertrophy (muscle growth) or Strength gain
- Column F - "Set_Number" (integer): The set number for a given exersize within the workout, a set number of 1 indicates that was my first set of a particular exersize that day
- Column G - "Weight" (float): How much weight in Lbs was used for the set, any exersize that was performed with dumbells will have the weight per dumbell while any other exersize will have the total weight that was used
- Column H - "Reps"(integer): How many repetitions were done in a given set
- Column I - "RPE"(integer): The Rate of Perceived Exertion for a given set measured on a scale from 1-10 with 10 meaning I could do 0 more reps, 9.5 meaning I could maybe do 1 more rep or add slightly more weight, 9 meaning I could do one more rep, 8.5 meaning I could for sure do 1 more rep maybe 2 more, and so forth
- Column J - "RPE_Norm_Weight" (float): Using an RPE scale and the reps and RPE of a given set this is the estimated one rep max I could have done on an exersize based on a given set, this allows us to compare strength trends across different rep ranges evenly
- Column K - "Body_Weight"(float): My body weight in Lbs the day the set was performed
- Column L - "Comment" (string): This is on a workout level not a set level meaning all sets performed in the same workout will share the same comment, this feild is used for my overall thoughts on the workout and how it went, any questions that involve factors that cannot be found in the other feilds of this table such as injuries, sentiment, mood, form comments, quality of pump, etc would be found in this comment column

"""

# ==========================================
# 2. PASSWORD PROTECTION
# ==========================================
st.set_page_config(page_title="Kevin Workout Assistant", layout="wide")

if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if not st.session_state.password_correct:
    st.title("Private Access")
    pwd = st.text_input("Enter Password", type="password")
    if st.button("Login"):
        if pwd == APP_PASSWORD:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Incorrect Password")
    st.stop()

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    
    if "gcp_service_account" in st.secrets:
        #For cloud creds
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    else:
        #For local creds
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
    
    client = gspread.authorize(creds)
    sheet = client.open_by_url(GOOGLE_SHEET_URL).worksheet("Cleaned_Data")
    return pd.DataFrame(sheet.get_all_records())

try:
    df = load_data()

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    # Create the Agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=SYSTEM_CONTEXT,
        agent_type="zero-shot-react-description",
    )

    # ==========================================
    # 4. CHAT INTERFACE
    # ==========================================
    st.title("Kevin Workout Assistant")
    
    # Sidebar data preview
    with st.sidebar:
        st.subheader("Data Preview")
        st.dataframe(df.head(5))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["type"] == "text":
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg["type"] == "chart":
            st.chat_message("assistant").pyplot(msg["content"])

    if prompt := st.chat_input("Ex: 'Draw a line chart of the last 30 days of Smith Incline Bench'"):
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing and generating visuals..."):
                try:
                    # Execute agent
                    response = agent.invoke(prompt)
                    answer = response["output"]
                    
                    # 1. Display text answer
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "type": "text"})
                    
                    # 2. Check if a chart was created in the background
                    fig = plt.gcf() # Get Current Figure
                    if fig.get_axes(): # If the figure has axes, it means a plot was drawn
                        st.pyplot(fig)
                        st.session_state.messages.append({"role": "assistant", "content": fig, "type": "chart"})
                        plt.clf() # Clear the figure so it doesn't duplicate next time
                        
                except Exception as e:
                    st.error(f"Error: {e}")

except Exception as e:
    st.error(f"Setup Error: {e}")