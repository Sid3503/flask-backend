import os
import requests
import re
from dotenv import load_dotenv


load_dotenv()
TEAM_API_KEY = os.getenv("TEAM_API_KEY")
from flask import Flask, request, jsonify
from flask_cors import CORS
from langdetect import detect
from aixplain.factories import ModelFactory, AgentFactory


GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Load AI models
doc_model = ModelFactory.get(os.getenv("DOC_MODEL_ID"))
summ_model = ModelFactory.get(os.getenv("SUMM_MODEL_ID"))
main_agent = AgentFactory.get(os.getenv("AGENT_MODEL_ID"))

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

def remove_markdown(text):
    """Removes markdown symbols and extra spaces from text."""
    text = re.sub(r'\*\*.*?\*\*', '', text)  # Remove headers (e.g., **Header**)
    text = re.sub(r'[\*\-] ', '', text)  # Remove bullet points
    text = re.sub(r'[#\*_\[\]()]', '', text)  # Remove markdown symbols
    text = re.sub(r'\n+', '\n', text).strip()  # Remove extra newlines
    return text

def format_text(text):
    """Formats text into structured paragraphs."""
    sections = text.split("\n")
    formatted_text = "\n\n".join(section.strip() for section in sections if section.strip())
    return formatted_text

def get_nearest_health_centers(latitude, longitude):
    """Fetch nearest public health centers using Google Places API"""
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius=5000&type=hospital&keyword=public%20health%20center&key={GOOGLE_MAPS_API_KEY}"
    
    response = requests.get(url)
    results = response.json().get("results", [])
    
    if not results:
        return {"error": "No health centers found nearby"}
    
    health_centers = [
        {
            "name": place["name"],
            "address": place.get("vicinity", "No address available"),
            "latitude": place["geometry"]["location"]["lat"],
            "longitude": place["geometry"]["location"]["lng"]
        }
        for place in results[:5]
    ]
    
    return health_centers

def get_route(start_lat, start_lon, end_lat, end_lon):
    """Generate a route using Google Directions API"""
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={start_lat},{start_lon}&destination={end_lat},{end_lon}&mode=driving&key={GOOGLE_MAPS_API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if "routes" not in data or not data["routes"]:
        return {"error": "No route found"}
    
    route = data["routes"][0]["overview_polyline"]["points"]
    return {"route_polyline": route}

@app.route("/ask", methods=["POST"])
def ask():
    """Handles AI health advice and summary generation."""
    try:
        data = request.json
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Detect language
        output_language = detect(question)

        # Get AI response
        formatted_query = f"{question} Response in {output_language}"
        agent_response = main_agent.run(formatted_query)
        formatted_response = agent_response["data"]["output"]
        form_response = remove_markdown(formatted_response)
        agent_answer = format_text(form_response)

        safe_response = agent_answer.replace("\n", " ")
        safe_response = safe_response.replace('"', '\\"').replace("'", "\\'")

        # Generate summary
        summ = summ_model.run({
            "question": question,
            "response": f"{safe_response}",
            "language": output_language
        })["data"]
        corrected_text = summ.encode('latin1').decode('utf-8')
        corr_text = remove_markdown(corrected_text)
        summary = format_text(corr_text)

        return jsonify({
            "response": agent_answer,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/doctors", methods=["POST"])
def find_doctors():
    """Handles doctor recommendations based on health condition and location."""
    try:
        data = request.json
        condition = data.get("condition", "")
        location = data.get("location", "")

        if not condition or not location:
            return jsonify({"error": "Condition and location required"}), 400

        # Fetch nearby doctors
        doctors = doc_model.run({
            "condition": condition,
            "location": location
        })["data"]

        return jsonify({"doctors": doctors})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health-centers", methods=["POST"])
def find_health_centers():
    """Finds nearest public health centers and generates a route"""
    try:
        data = request.json
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if not latitude or not longitude:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        
        # Get nearest health centers
        health_centers = get_nearest_health_centers(latitude, longitude)
        if "error" in health_centers:
            return jsonify(health_centers), 400
        
        # Get route to the first health center
        first_center = health_centers[0]
        route = get_route(latitude, longitude, first_center["latitude"], first_center["longitude"])

        return jsonify({
            "nearest_health_centers": health_centers,
            "route": route
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
