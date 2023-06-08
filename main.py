from transformers import pipeline

genius = pipeline("text2text-generation", model='beyond/genius-base-chinese')

sketch = "[MASK] 辛亥革命 [MASK] 清政府 [MASK]"

generated_text = genius(sketch, num_beams=3, do_sample=True, max_length=200)[0]['generated_text']
print(generated_text)
