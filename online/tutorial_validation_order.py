from rapidata import RapidataClient

rapi = RapidataClient()

validation_set = rapi.validation.create_compare_set(
     name="Example Compare Validation Set",
     instruction="Which of the AI generated images looks more realistic?",
     datapoints=[["https://assets.rapidata.ai/bad_ai_generated_image.png", 
         "https://assets.rapidata.ai/good_ai_generated_image.png"]], 
     truths=["https://assets.rapidata.ai/good_ai_generated_image.png"] 
)

# find the validation set by name
validation_set = rapi.validation.find_validation_sets("Example Compare Validation Set")[0] 

# or by id
validation_set = rapi.validation.get_validation_set_by_id("validation_set_id")

order = rapi.order.create_compare_order(
     name="Example Compare Validation Set",
     instruction="Which of the AI generated images looks more realistic?", 
     datapoints=[["https://assets.rapidata.ai/dalle-3_human.jpg", 
        "https://assets.rapidata.ai/flux_human.jpg"]],
     validation_set_id=validation_set.id
).run()

order.display_progress_bar()
results = order.get_results()