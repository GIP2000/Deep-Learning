import openai
import json
import os

openai.api_key = os.getenv("OPEN_AI_API_KEY")

class LabelsWithDescriptors:

    def __init__(self, index, labels, descriptors=None):
        self.index = index
        self.labels = labels
        if descriptors is not None:
            self.descriptors = descriptors
        else:
            self.descriptors = self.get_parsed_response(self.labels[0])

    @staticmethod
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
     photo:
-"""

    @staticmethod
    def get_response(input: str):
        return openai.Completion.create(
          model="text-davinci-002",
          prompt=LabelsWithDescriptors.build_input(input),
          temperature=0.7,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

    @staticmethod
    def get_parsed_response(input:str):
        raw = LabelsWithDescriptors.get_response(input).get("choices")[0]["text"]
        return [x[1:].strip() for x in raw.split("\n")]


def main(f):
    cats = json.load(f)['cats']
    newCats = []
    for (i, labels) in enumerate(cats):
        if i >= 10:
            break
        print(f"#{i}: {labels[0]}")
        newCats.append(LabelsWithDescriptors(i,labels))
    # print(json.dumps([x.__dict__ for x in newCats]))
    with open("new_cats.json", "w") as outfile:
        json.dump([x.__dict__ for x in newCats], outfile)



if __name__ == "__main__":
    with open("ImageNetClassifers.json") as f:
        main(f)

