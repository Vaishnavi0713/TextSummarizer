import torch
import gradio as gr

from transformers import pipeline

model_path = r"D:\Projects\models\distilbart-cnn-12-6"

text_summary = pipeline(
    "summarization",
    model=model_path
)

# text = "Artificial Intelligence (AI) is one of the most important technologies in the modern world. It enables machines and computers to perform tasks that normally require human intelligence, such as learning, problem-solving, decision-making, and understanding languages. AI is widely used in healthcare, education, transportation, banking, and agriculture. In healthcare, AI helps doctors detect diseases and analyze medical reports. In education, it provides personalized learning experiences for students. Self-driving cars use AI to recognize objects and make driving decisions. Businesses also use AI to automate repetitive tasks, improve customer service, and analyze large amounts of data. Although AI offers many advantages, it also creates challenges related to privacy, security, and job displacement. Therefore, AI should be developed and used responsibly to ensure that it benefits society."
#
# print(text_summary(text))


def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

demo = gr.Interface(fn=summary, inputs=[gr.Textbox(label = "Input text to summarize",lines = 6)], outputs=[gr.Textbox(label = "Summarized Text",lines = 4)], title ="TEXT SUMMARIZER", description ="This application will be used to summarize the text")
demo.launch()