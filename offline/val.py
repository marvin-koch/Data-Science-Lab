from rapidata import RapidataClient

rapi = RapidataClient()

validation_set = rapi.validation.create_compare_set(
     name="Example",
     instruction="Which of the AI generated images looks more realistic?",
     datapoints=[["https://assets.rapidata.ai/bad_ai_generated_image.png", 
         "https://assets.rapidata.ai/good_ai_generated_image.png"]], 
     truths=["https://assets.rapidata.ai/good_ai_generated_image.png"] 
)
