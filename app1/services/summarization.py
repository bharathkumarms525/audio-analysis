import os
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

def generate_summary_with_groq(text):
    """
    Generate a summary using Groq's Chat Completions API.
    :param text: Input text to summarize.
    :return: Summary as a string.
    """
    try:
       
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"

       
        response = client.chat.completions.create(
            model="llama3-8b-8192",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )

        # Extract and return the summary from the response
        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        print(f"Error using Groq model: {e}")
        return "Error generating summary"


