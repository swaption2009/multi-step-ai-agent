from google.cloud import discoveryengine_v1 as discoveryengine
from vertexai.generative_models import GenerativeModel
from config import PROJECT_NUMBER, SEARCH_ENGINE_ID
from typing import List
import json

#variables
spec = discoveryengine.GenerateGroundedContentRequest.GenerationSpec(
        model_id="gemini-1.5-flash",
        temperature=0.0,
        top_p=1,
        top_k=1,
)

#client initialisation
google_search_client = discoveryengine.GroundedGenerationServiceClient()
model = GenerativeModel(
    "gemini-1.5-pro",
    system_instruction=f"""You are Falcon, one of the most seasoned equity traders in the world.
        Your goal is to help to answer the user question with comprehensive analysis based on what you have been trained on, or knowledge from Google Search or knowledge from internal proprietary investment research.
        You need to return a response that explains how you came up with that answer, backed by evidence that you used in coming up with the answer.
        The user is a day trader, risk tolerance is high. time horizon for trading is usually 1-2 months. Investment goal is to maximise the opportunity cost of the funds and reap maximum returns within the time horizon.
        """
)


def google_ground(prompt: str) -> str:
    request = discoveryengine.GenerateGroundedContentRequest(
        location=google_search_client.common_location_path(
                project=PROJECT_NUMBER, location="global"
        ),
        generation_spec=spec,
        contents=[
            discoveryengine.GroundedGenerationContent(
                    role="user",
                    parts=[discoveryengine.GroundedGenerationContent.Part(text=prompt)],
            )
        ],
        system_instruction=discoveryengine.GroundedGenerationContent(
            parts=[
                discoveryengine.GroundedGenerationContent.Part(text="If given a stock or option to analyse, try to find relevant news from google search to see if it's a good time to buy more or sell. Use these news to formulate your analysis and return a comprehensive response targted to the provided stock or option. Rmb that you are a seasoned investment analyst so always remove any disclaimers about this not being financial advice.")
            ],
        ),
        grounding_spec=discoveryengine.GenerateGroundedContentRequest.GroundingSpec(
            grounding_sources=[
                discoveryengine.GenerateGroundedContentRequest.GroundingSource(
                        google_search_source=discoveryengine.GenerateGroundedContentRequest.GroundingSource.GoogleSearchSource()
                )
            ]
        ),
    )
    google_responses = google_search_client.generate_grounded_content(request)
    
    print(google_responses)  # Print the extracted text
    return_prompt=f"""Generate a natural language response based on the original question: '{prompt}' and the returned results: '{google_responses}'"""

    response=model.generate_content(return_prompt)
    return response.text

