import torch
import gradio as gr
from transformers import pipeline
import json

# model_path = "C:\\Users\\abdul\\Documents\\genaiproj\\genai\\Models\models--facebook--nllb-200-distilled-600M\\snapshots\\f8d333a098d19b4fd9a8b18f94170487ad3f821d"
text_translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", torch_dtype=torch.bfloat16)
# text_translator = pipeline("translation", model=model_path, torch_dtype=torch.bfloat16)

# with open('C:\\Users\\abdul\\Documents\\genaiproj\\genai\\Files\\language.json', 'r') as file:
with open('language.json', 'r') as file:

    language_data = json.load(file)

def get_flores_code_from_language(language_name):
    for entry in language_data:
        if entry["Language"].lower() == language_name.lower():
            return entry["FLORES-200 code"]
    return "Language not found."

def translate_text(text, destination_language):
    # text = input("Enter the text to translate: ")
    dest_code = get_flores_code_from_language(destination_language)
    translation = text_translator(text,
                              src_lang ="eng_Latn",
                              tgt_lang =dest_code)
    return translation[0]["translation_text"]


gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")

demo = gr.Interface(
    fn=translate_text, 
    inputs=[gr.Textbox(label="Input text to translate", lines=6), gr.Dropdown(["German", "French", "Tamil", "Romanian", "Arabic"], label="Select Destination Language")], 
    outputs=[gr.Textbox(label="Translated text", lines=4)], 
    title="Multilanguage Translator", 
    theme="soft",
    description="Translate text to any language in seconds!")
    
demo.launch(share=True)