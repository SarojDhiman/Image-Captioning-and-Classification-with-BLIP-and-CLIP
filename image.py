import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch

# Initialize BLIP Model for Image Captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize CLIP Model for Zero-Shot Classification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Categories for classification
categories = [
    "Unauthorized Property Usage - Property not owned or entitled to by the user",  # 3(1)(b)(i)
    "Inappropriate or Harmful Content - Includes pornographic, paedophilic, privacy violations, insulting, harassing, racially/ethnically objectionable content, promoting enmity, or inciting violence",  # 3(1)(b)(ii)
    "Child Harm - Harmful to children",  # 3(1)(b)(iii)
    "Intellectual Property Violation - Violates intellectual property rights",  # 3(1)(b)(iv)
    "False Information or Misleading Content - Misleading origin or communicates falsehoods",  # 3(1)(b)(v)
    "Impersonation - Impersonates another person",  # 3(1)(b)(vi)
    "National Security Violation - Undermines India's security",  # 3(1)(b)(vii)
    "Harmful Software Distribution - Includes harmful software",  # 3(1)(b)(viii)
    "Unverified Online Games - Unverified online games",  # 3(1)(b)(ix)
    "Unauthorized Game Promotion - Promotes unauthorized games",  # 3(1)(b)(x)
    "Violation of Existing Laws - Violates any law in force"  # 3(1)(b)(xi)
]

# categories = [
#     "This image is safe and contains no harmful content.",  # safe
#     "This image contains not safe for work (NSFW) material.",  # nsfw
#     "This image shows abusive behavior or language.",  # abusive
#     "This image depicts erotic or sexual content.",  # erotic
#     "This image contains violent actions or events.",  # violent
#     "This image contains hate speech or symbols.",  # hate speech
#     "This image contains graphic or disturbing content.",  # graphic content
#     "This image contains explicit nudity.",  # explicit nudity
#     "This image contains suggestive or provocative content.",  # suggestive content
#     "This image shows bullying or harassment.",  # bullying
#     "This image depicts harassment in some form.",  # harassment
#     "This image is related to terrorism or terroristic acts.",  # terrorism
#     "This image shows signs of extremism or extreme ideology.",  # extremism
#     "This image contains drug use or drug-related content.",  # drug use
#     "This image shows signs of self-harm or suicidal behavior.",  # self-harm
#     "This image contains spam or irrelevant content.",  # spam
#     "This image contains misinformation or fake news.",  # misinformation
#     "This image shows trolling or disruptive behavior.",  # trolling
#     "This image contains racist content or discrimination.",  # racist content
#     "This image involves child exploitation or abuse.",  # child exploitation
#     "This image shows animal abuse or cruelty.",  # animal abuse
#     "This image depicts violent extremism.",  # violent extremism
#     "This image shows cultural insensitivity or offensive behavior.",  # cultural insensitivity
#     "This image contains profanity or offensive language.",  # profanity
#     "This image shows graphic injuries or severe physical harm.",  # graphic injury
#     "This image involves illegal activities."  # illegal activities
# ]


def get_image_caption_blip(image):
    inputs = blip_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        output = blip_model.generate(**inputs)
        
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

def classify_image_clip(image, categories):
    # Prepare the inputs for CLIP model
    inputs = clip_processor(text=categories, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
    predicted_idx = probs.argmax().item()
    
    # Return the predicted category from the list
    return categories[predicted_idx]

def get_external_api_classification(image_url):
    # Placeholder function for external API call
    api_response = {"predicted_category": "graphic content"}  # Simulated response
    return api_response['predicted_category']

def ensemble_predict(image, image_url):
    predictions = []
    
    # BLIP model for detailed captioning
    caption = get_image_caption_blip(image)
    predictions.append(caption)

    # CLIP for zero-shot category classification
    clip_prediction = classify_image_clip(image, categories)
    predictions.append(clip_prediction)

    # External API for content moderation
    api_prediction = get_external_api_classification(image_url)
    predictions.append(api_prediction)
    
    # Voting mechanism to decide final category
    final_prediction = max(set(predictions), key=predictions.count)
    
    return caption, final_prediction

# Example usage with an image URL
# url = "https://pulitzercenter.org/sites/default/files/styles/768x600/public/04-01-13/MineIsBorn_Untold_Stories_Price_Burkina_Faso02_0.jpg.webp?itok=U1kxrDz9"
# url = "https://assets.nst.com.my/images/articles/17ntbudak.jpg_1530057915.jpg"
# url = "https://theirworld.org/wp-content/uploads/2017/05/Child-labour-3.jpg"
# url = "https://images.unsplash.com/photo-1557089041-7fa93ffc2e08?q=80&w=1769&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# url = "https://images.unsplash.com/photo-1557089041-7fa93ffc2e08?q=80&w=1769&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# url = "https://i0.wp.com/world-education-blog.org/wp-content/uploads/2016/11/11.jpg?fit=458%2C418&ssl=1"
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe3nB9TY5DrNyt1Ia4i1mNLTUHVX25dv44gA&s"
# url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Street_Theatre_on_Domestic_Violence_-_Bridge_Market_Plaza_-_Chandigarh_2016-08-07_9101.JPG/1200px-Street_Theatre_on_Domestic_Violence_-_Bridge_Market_Plaza_-_Chandigarh_2016-08-07_9101.JPG"
# url = "https://content.presspage.com/uploads/1065/500_abuseis6846493medium-2.jpg"
# url = "https://img.freepik.com/free-photo/bad-sex-concept-with-upset-couple_23-2149070988.jpg"
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT_1OBObO4i2Xgx3o56NNyygpw3ahKWw53XDsBVV9xm492o9uBR_5AMs0HICgGW0L9Q4rE&usqp=CAU"
# url = "https://qph.cf2.quoracdn.net/main-qimg-e47461bf5cfb32b985c7216c2b51785c-lq"
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZomD3mOSrIYO7sSlDtkmzOFwufxNTV3wSOTTDIuzgBXpyekmwXM-M2YDxY4EOlUhPdLg&usqp=CAU"
# url = "https://www.psychologs.com/wp-content/uploads/2024/02/The-Psychology-of-Unethical-Behavior.jpg"
# url = "https://static.toiimg.com/thumb/imgsize-23456,msid-100833585,width-600,resizemode-4/100833585.jpg"
url = "https://assets3.cbsnewsstatic.com/hub/i/r/2011/12/13/55a51375-a645-11e2-a3f0-029118418759/thumbnail/640x480/8537fd6c805702df58056a18ec9bb7cf/toddlerpilloverdose_0000042.jpg?v=29ebd300d9a3cd24077d945a46991f72"
# url = "https://www.hrw.org/sites/default/files/styles/embed_xxl/public/media_2021/02/202102asia_india_delhi_riots.JPG?itok=aVEObnl8"

response = requests.get(url, stream=True)

# Check if the image was correctly downloaded
if response.status_code == 200 and 'image' in response.headers['Content-Type']:
    image = Image.open(BytesIO(response.content))
    
    # Run the ensemble prediction
    caption, predicted_category = ensemble_predict(image, url)
    print(f"Image Caption: {caption}")
    print(f"Predicted Category: {predicted_category}")
else:
    print("Failed to retrieve a valid image.")


# from transformers import BlipProcessor, BlipForConditionalGeneration
# from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltForQuestionAnswering
# import requests
# from PIL import Image
# from io import BytesIO
# import torch

# # Initialize BLIP Model for Image Captioning

# # Initialize Models
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # Categories for classification
# categories = [
#     "safe", 
#     "nsfw", 
#     "abusive",
#     "erotic",
#     "violent", 
#     "hate speech", 
#     "graphic content",
#     "explicit nudity",
#     "suggestive content",
#     "bullying",
#     "harassment",
#     "terrorism",
#     "extremism",
#     "drug use",
#     "self-harm",
#     "spam",
#     "misinformation",
#     "trolling",
#     "racist content",
#     "child exploitation",
#     "animal abuse",
#     "violent extremism",
#     "cultural insensitivity",
#     "profanity",
#     "graphic injury",
#     "illegal activities"
# ]

# def get_image_caption_blip(image):
#     inputs = blip_processor(images=image, return_tensors="pt")
    
#     with torch.no_grad():
#         output = blip_model.generate(**inputs)
        
#     caption = blip_processor.decode(output[0], skip_special_tokens=True)
#     return caption

# def classify_image_clip(image, categories):
#     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
#     inputs = clip_processor(text=categories, images=image, return_tensors="pt", padding=True)
#     outputs = clip_model(**inputs)
    
#     logits_per_image = outputs.logits_per_image
#     probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
#     predicted_idx = probs.argmax().item()
#     return categories[predicted_idx]

# def get_external_api_classification(image_url):
#     # Placeholder function for external API call
#     api_response = {"predicted_category": "graphic content"}  # Simulated response
#     return api_response['predicted_category']

# def ensemble_predict(image, image_url):
#     predictions = []
    
#     # BLIP model for detailed captioning
#     caption = get_image_caption_blip(image)
#     predictions.append(caption)

#     # CLIP for zero-shot category classification
#     clip_prediction = classify_image_clip(image, categories)
#     predictions.append(clip_prediction)

#     # External API for content moderation
#     api_prediction = get_external_api_classification(image_url)
#     predictions.append(api_prediction)
    
#     # Voting mechanism to decide final category
#     final_prediction = max(set(predictions), key=predictions.count)
    
#     return caption, final_prediction

# # Example usage with an image URL
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe3nB9TY5DrNyt1Ia4i1mNLTUHVX25dv44gA&s"

# response = requests.get(url, stream=True)

# # Check if the image was correctly downloaded
# if response.status_code == 200 and 'image' in response.headers['Content-Type']:
#     image = Image.open(BytesIO(response.content))
    
#     # Run the ensemble prediction
#     caption, predicted_category = ensemble_predict(image, url)
#     print(f"Image Caption: {caption}")
#     print(f"Predicted Category: {predicted_category}")
# else:
#     print("Failed to retrieve a valid image.")


# import requests
# from PIL import Image
# from io import BytesIO
# from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltForQuestionAnswering
# import torch

# # Initialize Models
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# # Categories for classification
# categories = [
#     "safe", 
#     "nsfw", 
#     "abusive",
#     "erotic",
#     "violent", 
#     "hate speech", 
#     "graphic content",
#     "explicit nudity",
#     "suggestive content",
#     "bullying",
#     "harassment",
#     "terrorism",
#     "extremism",
#     "drug use",
#     "self-harm",
#     "spam",
#     "misinformation",
#     "trolling",
#     "racist content",
#     "child exploitation",
#     "animal abuse",
#     "violent extremism",
#     "cultural insensitivity",
#     "profanity",
#     "graphic injury",
#     "illegal activities"
# ]

# def get_image_caption_vilt(image):
#     text = "What does this image depict?"
#     inputs = vilt_processor(image, text, return_tensors="pt")
    
#     with torch.no_grad():
#         outputs = vilt_model(**inputs)

#     logits = outputs.logits
#     idx = logits.argmax(-1).item()
#     return vilt_model.config.id2label[idx]

# def classify_image_clip(image, categories):
#     inputs = clip_processor(text=categories, images=image, return_tensors="pt", padding=True)
#     outputs = clip_model(**inputs)
    
#     logits_per_image = outputs.logits_per_image
#     probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
#     predicted_idx = probs.argmax().item()
#     return categories[predicted_idx]

# def get_external_api_classification(image_url):
#     # Placeholder function for external API call
#     api_response = {"predicted_category": "graphic content"}  # Simulated response
#     return api_response['predicted_category']

# def ensemble_predict(image, image_url):
#     predictions = []
    
#     # ViLT model for caption and specific violence-related queries
#     caption = get_image_caption_vilt(image)
#     predictions.append(caption)

#     # CLIP for zero-shot category classification
#     clip_prediction = classify_image_clip(image, categories)
#     predictions.append(clip_prediction)

#     # External API for content moderation
#     api_prediction = get_external_api_classification(image_url)
#     predictions.append(api_prediction)
    
#     # Voting mechanism to decide final category
#     final_prediction = max(set(predictions), key=predictions.count)
    
#     return caption, final_prediction

# # Example usage with an image URL
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe3nB9TY5DrNyt1Ia4i1mNLTUHVX25dv44gA&s"

# response = requests.get(url, stream=True)

# # Check if the image was correctly downloaded
# if response.status_code == 200 and 'image' in response.headers['Content-Type']:
#     image = Image.open(BytesIO(response.content))
    
#     # Run the ensemble prediction
#     caption, predicted_category = ensemble_predict(image, url)
#     print(f"Image Caption: {caption}")
#     print(f"Predicted Category: {predicted_category}")
# else:
#     print("Failed to retrieve a valid image.")



# from transformers import ViltProcessor, ViltForQuestionAnswering

# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# import requests
# from PIL import Image

# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# # download an input image
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe3nB9TY5DrNyt1Ia4i1mNLTUHVX25dv44gA&s"
# # url = "https://content.presspage.com/uploads/1065/500_abuseis6846493medium-2.jpg"
# # url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Street_Theatre_on_Domestic_Violence_-_Bridge_Market_Plaza_-_Chandigarh_2016-08-07_9101.JPG/1200px-Street_Theatre_on_Domestic_Violence_-_Bridge_Market_Plaza_-_Chandigarh_2016-08-07_9101.JPG"
# image = Image.open(requests.get(url, stream=True).raw)
# text = "Is there any violence in the image?"

# # prepare inputs
# inputs = processor(image, text, return_tensors="pt")
# import torch

# # forward pass
# with torch.no_grad():
#     outputs = model(**inputs)

# logits = outputs.logits
# idx = logits.argmax(-1).item()
# print("Predicted answer:", model.config.id2label[idx])
