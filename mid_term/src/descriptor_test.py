import openai
import json
import os
from sys import argv

openai.api_key = os.getenv("OPEN_AI_API_KEY")

class LabelsWithDescriptors:


    def __init__(self, index, labels, descriptors=None):
        self.index = index
        self.labels = labels
        if descriptors is not None:
            self.descriptors = descriptors
        else:
            self.descriptors = LabelsWithDescriptors._get_parsed_response(self.labels[0])


    @staticmethod
    def _build_input(category_name: str) -> str:
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
    def _get_response(input: str):
        return openai.Completion.create(
          model="text-davinci-002",
          prompt=LabelsWithDescriptors._build_input(input),
          temperature=0.7,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )


    @staticmethod
    def _get_parsed_response(input:str):
        raw = LabelsWithDescriptors._get_response(input).get("choices")[0]["text"]
        return [x[1:].strip() for x in raw.split("\n")]


    @staticmethod
    def read_list_from_file(file_path: str):
        with open(file_path, 'r') as f:
            pass



def create_descriptors_from_file(f):
    cats = json.load(f)['cats']
    newCats = []
    for (i, labels) in enumerate(cats):
        if i >= 10:
            break
        print(f"#{i}: {labels[0]}")
        newCats.append(LabelsWithDescriptors(i,labels))

    with open("new_cats.json", "w") as outfile:
        json.dump([x.__dict__ for x in newCats], outfile)



if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python descriptor_test.py -[g|r] [file_path]")
        exit(1)
    elif argv[1] != "-g" and argv[1] != "-r":
        print("Usage: python descriptor_test.py -[g|r] [file_path]")
        exit(2)

    [_, flag, path] = argv
    if flag == "-g":
        with open(path) as f:
            create_descriptors_from_file(f)
        exit(0)
    LabelsWithDescriptors.read_list_from_file(path)
    exit(0)
