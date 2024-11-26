## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:
Develop an accessible and user-friendly platform for generating high-quality images from text prompts using the Stable Diffusion model. The application must allow real-time user interaction, customizable settings for image generation, and facilitate evaluation and feedback from users
### DESIGN STEPS:

#### STEP 1:
Install the required libraries: transformers, diffusers, torch, and gradio.
Load the Stable Diffusion model using the diffusers library.

#### STEP 2:
Create a Python function to take a text prompt as input and generate an image using the model.
Allow optional parameters like image resolution, number of inference steps, and guidance scale for flexibility.
#### STEP 3:
Design a Gradio interface to accept user input and display generated images.
Configure sliders or text boxes for user-controlled parameters (e.g., resolution, style).
Deploy the application on a local server or a cloud platform.
### PROGRAM:
```

# Import required libraries
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Step 1: Load the Stable Diffusion model
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

# Initialize the pipeline
pipe = load_model()

# Step 2: Define the image generation function
def generate_image(prompt, num_inference_steps=50, guidance_scale=7.5):
    """
    Generates an image based on the text prompt using Stable Diffusion.
    """
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

# Step 3: Set up Gradio Interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Stable Diffusion Image Generator")
        
        # Input elements
        prompt = gr.Textbox(label="Enter your prompt", placeholder="Describe the image you'd like to generate")
        num_steps = gr.Slider(10, 100, value=50, step=1, label="Number of Inference Steps")
        guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
        
        # Output element
        output_image = gr.Image(label="Generated Image")
        
        # Button to generate image
        generate_btn = gr.Button("Generate Image")
        
        # Define button behavior
        generate_btn.click(fn=generate_image, inputs=[prompt, num_steps, guidance], outputs=output_image)
    
    # Launch the app
    demo.launch()

# Run the Gradio app
if __name__ == "__main__":
    main()
```

### OUTPUT:
![Screenshot 2024-11-26 124600](https://github.com/user-attachments/assets/44a806a5-db82-438a-a841-e1dbf4273497)

### RESULT:
The prototype successfully demonstrates text-to-image generation using Stable Diffusion, providing an interactive and user-friendly interface. Users can modify parameters to influence the output quality and style.
