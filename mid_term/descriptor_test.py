import os
import openai

openai.api_key = "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"

def build_input(category_name: str) -> str:
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a
     photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a
     photo
-"""



response = openai.Completion.create(
  model="text-davinci-002",
  prompt=build_input("truck"),
  temperature=0.7,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response)
