import pandas as pd
import numpy as np
import streamlit as st
import datetime
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json
import os
from datetime import datetime, timedelta
import random


st.set_page_config(
    page_title="FemHealth: Personalized Cycle Tracking",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .stApp {
        background-color: #E2787B;
        color: white;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    h1, h2, h3 {
        color: #white !important;
    }
    
    .stButton button {
        background-color: #ff80ab !important;
        color: white !important;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #E2787B !important;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_USER_DATA = {
    'name': 'User',
    'age': 25,
    'has_pcos': False,
    'cycle_data': {
        'average_cycle_length': 28,
        'last_period_start': datetime.now().strftime('%Y-%m-%d'),
        'periods': [],
        'symptoms': {}
    },
    'health_metrics': {
        'weight': [],
        'sleep_hours': [],
        'stress_level': []
    },
    'medications': [],
    'doctor_appointments': [],
    'chat_history': []
}


def save_user_data(user_data):
    """Save user data to a JSON file."""
    try:
        with open('user_data.json', 'w') as f:
            json.dump(user_data, f, default=str)
    except Exception as e:
        st.error(f"Error saving user data: {e}")

def load_user_data():
    """Load user data from JSON file."""
    try:
        if os.path.exists('user_data.json'):
            with open('user_data.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading user data: {e}")
    return DEFAULT_USER_DATA


def predict_next_period(last_period_start, average_cycle_length):
    """
    Predict the next period start and end dates
    
    Args:
    - last_period_start (str): Date of last period start
    - average_cycle_length (int): Average length of menstrual cycle
    
    Returns:
    - Predicted next period start and end dates
    """
    try:
        last_period = datetime.strptime(last_period_start, '%Y-%m-%d')
        next_period_start = last_period + timedelta(days=average_cycle_length)
        next_period_end = next_period_start + timedelta(days=5)  # Assuming 5-day period
        
        return {
            'predicted_start': next_period_start.strftime('%Y-%m-%d'),
            'predicted_end': next_period_end.strftime('%Y-%m-%d')
        }
    except Exception as e:
        st.error(f"Error predicting period: {e}")
        return None

# Symptom Management Database
SYMPTOM_REMEDIES = {
    "cramps": [
        "Apply a heating pad to your lower abdomen",
        "Practice gentle yoga or stretching exercises",
        "Try over-the-counter pain relievers like ibuprofen",
        "Stay hydrated and drink warm beverages like ginger tea",
        "Consider supplements like magnesium or vitamin B1"
    ],
    "bloating": [
        "Reduce salt intake during your period",
        "Avoid carbonated beverages and foods that cause gas",
        "Exercise regularly to reduce water retention",
        "Try natural diuretics like cucumber, watermelon, or dandelion tea",
        "Consider wearing loose, comfortable clothing"
    ],
    "fatigue": [
        "Prioritize getting 7-9 hours of sleep each night",
        "Incorporate iron-rich foods like spinach and lean meats in your diet",
        "Stay hydrated throughout the day",
        "Try low-intensity exercises like walking or swimming",
        "Consider taking short power naps (20-30 minutes) when needed"
    ],
    "headache": [
        "Practice relaxation techniques like deep breathing",
        "Apply a cold or warm compress to your forehead or neck",
        "Maintain regular sleep patterns and stay hydrated",
        "Try over-the-counter pain relievers",
        "Consider reducing screen time and bright light exposure"
    ],
    "mood swings": [
        "Practice mindfulness meditation or deep breathing exercises",
        "Get regular physical activity to boost endorphins",
        "Maintain a consistent sleep schedule",
        "Try supplements like vitamin B6 or calcium",
        "Consider speaking with a healthcare provider about serious mood issues"
    ],
    "acne": [
        "Keep your face clean with a gentle cleanser twice daily",
        "Avoid touching your face throughout the day",
        "Use non-comedogenic skincare products",
        "Try spot treatments with salicylic acid or benzoyl peroxide",
        "Consider dietary changes like reducing dairy and sugar intake"
    ]
}

# Initialize Gemini API
def initialize_gemini_api(api_key):
    """Initialize Google Generative AI."""
    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {e}")
        return False

def initialize_session_state():
    """Initialize session state variables."""
 
    default_states = {
        'page': 'dashboard',
        'gemini_api_key': None,
        'user_data': load_user_data()
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


def navigate_to(page):
    """Navigate between pages."""
    st.session_state.page = page
    st.rerun()


def add_sidebar_navigation():
    """Add sidebar navigation options."""
    st.sidebar.header("Navigation")
    if st.sidebar.button("Dashboard"):
        navigate_to("dashboard")
    if st.sidebar.button("Track Period"):
        navigate_to("track_period")
    if st.sidebar.button("Track Symptoms"):
        navigate_to("track_symptoms")
    if st.sidebar.button("Health Metrics"):
        navigate_to("health_metrics")
    if st.sidebar.button("AI Assistant"):
        navigate_to("ai_assistant")
    
   
    st.sidebar.header("🔑 Gemini API Key")
    api_key = st.sidebar.text_input(
        "Enter your Gemini API Key", 
        type="password", 
        value=st.session_state.get('gemini_api_key', ''),
        help="Get your API key from Google AI Studio"
    )
    
    if api_key:
        if initialize_gemini_api(api_key):
            st.session_state.gemini_api_key = api_key
            st.sidebar.success("API Key Validated Successfully!")
        else:
            st.session_state.gemini_api_key = None


def dashboard_page():
    """Render user dashboard."""
    user_data = st.session_state.user_data
    
    st.title(f"Welcome, {user_data['name']}! 👋")

    st.subheader("Your Next Period Prediction")
    last_period = user_data['cycle_data']['last_period_start']
    avg_cycle_length = user_data['cycle_data'].get('average_cycle_length', 28)
    
    prediction = predict_next_period(last_period, avg_cycle_length)
    if prediction:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Next Period Start", prediction['predicted_start'])
        with col2:
            st.metric("Next Period End", prediction['predicted_end'])
            
 
    if user_data['cycle_data']['symptoms']:
        st.subheader("Recent Symptoms")
        symptoms_list = []
        for symptom, dates in user_data['cycle_data']['symptoms'].items():
            if dates: 
                symptoms_list.append(f"{symptom.capitalize()} (last recorded: {dates[-1]})")
        
        if symptoms_list:
            for symptom in symptoms_list[:3]:  
                st.write(f"• {symptom}")
            
            if len(symptoms_list) > 3:
                st.write(f"... and {len(symptoms_list) - 3} more")
                
            st.button("Manage Symptoms", on_click=lambda: navigate_to("track_symptoms"))

    add_sidebar_navigation()

# Period Tracking Page
def track_period_page():
    """Render period tracking page."""
    st.title("Period Tracking")
    
    user_data = st.session_state.user_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Log Your Period")
        period_start = st.date_input("Start Date of Period")
        period_end = st.date_input("End Date of Period", value=period_start + timedelta(days=5))
        flow_intensity = st.selectbox("Flow Intensity", ["Light", "Medium", "Heavy"])
        
        if st.button("Save Period Log"):
            new_period_entry = {
                'start_date': period_start.strftime('%Y-%m-%d'),
                'end_date': period_end.strftime('%Y-%m-%d'),
                'flow': flow_intensity.lower(),
                'symptoms': [],
                'mood': ''
            }
            
            # Update last period start and periods list
            user_data['cycle_data']['last_period_start'] = period_start.strftime('%Y-%m-%d')
            user_data['cycle_data']['periods'].append(new_period_entry)
            
            # Recalculate average cycle length
            if len(user_data['cycle_data']['periods']) > 1:
                cycle_lengths = []
                for i in range(1, len(user_data['cycle_data']['periods'])):
                    prev_period = datetime.strptime(user_data['cycle_data']['periods'][i-1]['start_date'], '%Y-%m-%d')
                    curr_period = datetime.strptime(user_data['cycle_data']['periods'][i]['start_date'], '%Y-%m-%d')
                    cycle_lengths.append((curr_period - prev_period).days)
                
                if cycle_lengths:
                    user_data['cycle_data']['average_cycle_length'] = round(sum(cycle_lengths) / len(cycle_lengths))
            
            # Save updated user data
            save_user_data(user_data)
            st.session_state.user_data = user_data
            
            st.success("Period log saved successfully!")
    
    with col2:
        # Show period history
        st.subheader("Period History")
        if user_data['cycle_data']['periods']:
            df = pd.DataFrame(user_data['cycle_data']['periods'])
            st.dataframe(df[['start_date', 'end_date', 'flow']])
        else:
            st.info("No period logs available. Start tracking by logging your period.")
    
    # Show next period prediction
    st.subheader("Period Prediction")
    last_period = user_data['cycle_data']['last_period_start']
    avg_cycle_length = user_data['cycle_data'].get('average_cycle_length', 28)
    
    prediction = predict_next_period(last_period, avg_cycle_length)
    if prediction:
        st.info(f"Based on your data, your next period is predicted from {prediction['predicted_start']} to {prediction['predicted_end']}")
        
        # Calculate days until next period
        today = datetime.now().date()
        next_period_date = datetime.strptime(prediction['predicted_start'], '%Y-%m-%d').date()
        days_until = (next_period_date - today).days
        
        if days_until > 0:
            st.metric("Days until next period", days_until)
        elif days_until == 0:
            st.metric("Days until next period", "Today!")
        else:
            st.metric("Current day of period", abs(days_until) + 1)

    # Add sidebar navigation
    add_sidebar_navigation()

# Symptom Tracking Page
def track_symptoms_page():
    """Render symptom tracking page."""
    st.title("Symptom Tracking & Management")
    
    user_data = st.session_state.user_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Log Your Symptoms")
        symptoms = ["Cramps", "Bloating", "Fatigue", "Headache", "Mood Swings", "Acne"]
        selected_symptoms = st.multiselect("Select Symptoms You're Currently Experiencing", symptoms)
        
        severity = st.slider("Symptom Severity (1-10)", min_value=1, max_value=10, value=5)
        notes = st.text_area("Additional Notes", height=100)
        
        if st.button("Log Symptoms"):
            today = datetime.now().strftime('%Y-%m-%d')
            
            for symptom in selected_symptoms:
                symptom_key = symptom.lower()
                if symptom_key not in user_data['cycle_data']['symptoms']:
                    user_data['cycle_data']['symptoms'][symptom_key] = []
                
                # Add today's date with severity and notes
                user_data['cycle_data']['symptoms'][symptom_key].append({
                    'date': today,
                    'severity': severity,
                    'notes': notes
                })
            
            # Save updated user data
            save_user_data(user_data)
            st.session_state.user_data = user_data
            
            st.success("Symptoms logged successfully!")
    
    with col2:
        st.subheader("Symptom Management")
        
        if selected_symptoms:
            st.write("Here are some remedies for your selected symptoms:")
            
            for symptom in selected_symptoms:
                symptom_key = symptom.lower()
                
                with st.expander(f"Remedies for {symptom}"):
                    if symptom_key in SYMPTOM_REMEDIES:
                        for i, remedy in enumerate(SYMPTOM_REMEDIES[symptom_key], 1):
                            st.write(f"{i}. {remedy}")
                    else:
                        st.write("No specific remedies available. Consider consulting with a healthcare provider.")
        else:
            st.write("Select symptoms above to see personalized remedies.")
            
            # AI-powered symptom management
            st.subheader("AI-Powered Advice")
            
            if selected_symptoms and st.session_state.get('gemini_api_key'):
                try:
                    # Create a prompt for the AI assistant
                    prompt = f"""
                    I'm experiencing {', '.join(selected_symptoms).lower()} with a severity of {severity}/10.
                    {notes if notes else ''}
                    
                    Please provide 3 scientifically-backed remedies or management strategies for these symptoms.
                    Format your response as a bullet list with brief explanations.
                    """
                    
                    # Call the AI for personalized advice
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    response = model.generate_content(prompt)
                    
                    st.write("**AI-Powered Personalized Advice:**")
                    st.write(response.text)
                    
                    # Add this advice to the chat history
                    user_data['chat_history'].append({'role': 'user', 'message': prompt})
                    user_data['chat_history'].append({'role': 'assistant', 'message': response.text})
                    save_user_data(user_data)
                    
                except Exception as e:
                    st.error(f"Could not generate AI advice: {e}")
                    st.info("Please check your Gemini API key in the sidebar.")
            elif selected_symptoms:
                st.info("Add a Gemini API key in the sidebar to get personalized AI advice.")
    
    # Display symptom history
    st.subheader("Symptom History")
    
    if user_data['cycle_data']['symptoms']:
        symptom_history = []
        
        for symptom, entries in user_data['cycle_data']['symptoms'].items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and 'date' in entry:
                        symptom_history.append({
                            'symptom': symptom.capitalize(),
                            'date': entry['date'],
                            'severity': entry.get('severity', 'N/A'),
                            'notes': entry.get('notes', '')
                        })
                    elif isinstance(entry, str):  # Handle old format where only dates were stored
                        symptom_history.append({
                            'symptom': symptom.capitalize(),
                            'date': entry,
                            'severity': 'N/A',
                            'notes': ''
                        })
        
        if symptom_history:
            df = pd.DataFrame(symptom_history)
            df = df.sort_values('date', ascending=False)
            st.dataframe(df)
        else:
            st.info("No symptom history available.")
    else:
        st.info("No symptom history available. Start tracking by logging your symptoms.")

    # Add sidebar navigation
    add_sidebar_navigation()

# Health Metrics Page
def health_metrics_page():
    """Render health metrics page."""
    st.title("Health Metrics")
    
    user_data = st.session_state.user_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Track Your Health Metrics")
        
        # Track weight
        weight_tab, sleep_tab, stress_tab = st.tabs(["Weight", "Sleep", "Stress"])
        
        with weight_tab:
            weight = st.number_input("Enter your weight (kg)", min_value=0.0, step=0.1)
            if st.button("Log Weight"):
                user_data['health_metrics']['weight'].append({
                    'date': datetime.now().strftime('%Y-%m-%d'), 
                    'value': weight
                })
                save_user_data(user_data)
                st.session_state.user_data = user_data
                st.success("Weight logged successfully!")
        
        # Track sleep hours
        with sleep_tab:
            sleep_hours = st.number_input("Enter your sleep hours", min_value=0.0, max_value=24.0, step=0.5)
            if st.button("Log Sleep Hours"):
                user_data['health_metrics']['sleep_hours'].append({
                    'date': datetime.now().strftime('%Y-%m-%d'), 
                    'value': sleep_hours
                })
                save_user_data(user_data)
                st.session_state.user_data = user_data
                st.success("Sleep hours logged successfully!")
        
        # Track stress level
        with stress_tab:
            stress_level = st.slider("Enter your stress level (1-10)", min_value=1, max_value=10)
            if st.button("Log Stress Level"):
                user_data['health_metrics']['stress_level'].append({
                    'date': datetime.now().strftime('%Y-%m-%d'), 
                    'value': stress_level
                })
                save_user_data(user_data)
                st.session_state.user_data = user_data
                st.success("Stress level logged successfully!")
    
    with col2:
        st.subheader("View Health Metrics")
        
        metric_type = st.selectbox("Select metric to view", ["Weight", "Sleep Hours", "Stress Level"])
        
        # Create dataframe for selected metric
        if metric_type == "Weight":
            df = pd.DataFrame(user_data['health_metrics']['weight'])
        elif metric_type == "Sleep Hours":
            df = pd.DataFrame(user_data['health_metrics']['sleep_hours'])
        elif metric_type == "Stress Level":
            df = pd.DataFrame(user_data['health_metrics']['stress_level'])
        
        # Display metric data
        if not df.empty:
            # Convert date strings to datetime objects
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create a line chart
            fig = px.line(df, x='date', y='value', title=f"{metric_type} Over Time")
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=metric_type,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig)
            
            # Show statistics
            st.subheader(f"{metric_type} Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average", round(df['value'].mean(), 2))
            with col2:
                st.metric("Minimum", round(df['value'].min(), 2))
            with col3:
                st.metric("Maximum", round(df['value'].max(), 2))
            
            # Show raw data
            with st.expander("View Raw Data"):
                st.dataframe(df)
        else:
            st.info(f"No {metric_type.lower()} data available yet. Start tracking by logging your {metric_type.lower()}.")
    
    # Add sidebar navigation
    add_sidebar_navigation()

# AI Assistant Page
def ai_assistant_page():
    """Render AI assistant page."""
    st.title("AI Assistant")
    
    user_data = st.session_state.user_data
    
    # Check if Gemini API key is set
    if not st.session_state.get('gemini_api_key'):
        st.warning("Please enter your Gemini API key in the sidebar to use the AI Assistant.")
        add_sidebar_navigation()
        return
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat with Your Health Assistant")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for chat in user_data['chat_history'][-10:]:  # Show last 10 messages
                role = "You" if chat['role'] == 'user' else "Assistant"
                
                if role == "You":
                    st.markdown(f"<div style='background-color:#2b2b2b; padding:10px; border-radius:10px; margin-bottom:10px;'><strong>{role}:</strong> {chat['message']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#3b3b3b; padding:10px; border-radius:10px; margin-bottom:10px;'><strong>{role}:</strong> {chat['message']}</div>", unsafe_allow_html=True)
        
        # Chat input
        st.subheader("Ask a Question")
        user_query = st.text_area("Type your question here:", height=100)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            send_button = st.button("Send", use_container_width=True)
        with col2:
            clear_button = st.button("Clear Chat", use_container_width=True)
        
        if send_button and user_query:
            try:
                # Enhance the prompt with context
                enhanced_prompt = f"""
                You are a women's health assistant. The user has the following health profile:
                - Age: {user_data['age']}
                - Has PCOS: {"Yes" if user_data.get('has_pcos', False) else "No or Unknown"}
                - Average cycle length: {user_data['cycle_data'].get('average_cycle_length', 28)} days
                
                The user asks: {user_query}
                
                Please provide a helpful, accurate, and compassionate response.
                """
                
                # Send the query to the Gemini API
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(enhanced_prompt)
                answer = response.text
                
                # Save the chat history
                user_data['chat_history'].append({'role': 'user', 'message': user_query})
                user_data['chat_history'].append({'role': 'assistant', 'message': answer})
                save_user_data(user_data)
                st.session_state.user_data = user_data
                
                st.rerun() # Refresh to show the new messages
            except Exception as e:
                st.error(f"Error communicating with AI Assistant: {e}")
        
        if clear_button:
            user_data['chat_history'] = []
            save_user_data(user_data)
            st.session_state.user_data = user_data
            st.rerun()
    
    with col2:
        st.subheader("Quick Questions")
        
        # Add quick question buttons
        quick_questions = [
            "What can help with period cramps?",
            "How does diet affect my cycle?",
            "What are signs of hormonal imbalance?",
            "How can I track my fertility?",
            "What should I know about PCOS?",
            "How can I improve my sleep during my period?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                try:
                    # Send the query to the Gemini API
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    response = model.generate_content(f"Answer this question about women's health concisely: {question}")
                    answer = response.text
                    
                    # Save the chat history
                    user_data['chat_history'].append({'role': 'user', 'message': question})
                    user_data['chat_history'].append({'role': 'assistant', 'message': answer})
                    save_user_data(user_data)
                    st.session_state.user_data = user_data
                    
                    st.rerun() 
                except Exception as e:
                    st.error(f"Error communicating with AI Assistant: {e}")
    

    add_sidebar_navigation()

def main():
    """Main application function."""
    initialize_session_state()

    # Routing for different pages
    if st.session_state.page == 'dashboard':
        dashboard_page()
    elif st.session_state.page == 'track_period':
        track_period_page()
    elif st.session_state.page == 'track_symptoms':
        track_symptoms_page()
    elif st.session_state.page == 'health_metrics':
        health_metrics_page()
    elif st.session_state.page == 'ai_assistant':
        ai_assistant_page()


if __name__ == "__main__":
    main()