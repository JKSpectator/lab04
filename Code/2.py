import requests

API_URL = ""
API_KEY = ""

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": "gpt-4o",
    "messages": [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I would like to model an AI panoramic system. The users mainly include panorama "
                                    "image collectors, AI maintainers, administrators, and general users. The system "
                                    "should allow users to log in to the website to view panoramic images and ask "
                                    "questions to the AI. Panorama image collectors should be able to gather "
                                    "panoramic images and upload them to the website's database. AI maintainers are "
                                    "responsible for deploying interactive AI, and administrators manage other users."}
    ]
}

response = requests.post(API_URL, json=data, headers=headers)
print(response.json())