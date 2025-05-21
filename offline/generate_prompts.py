import random
import os
from itertools import product

# # Define all prompt components
# industries = [
#     "tech startup", "coffee shop", "fitness brand", "eco-friendly company", "law firm", "fashion brand",
#     "gaming company", "music festival", "non-profit organization", "finance app", "education platform",
#     "medical clinic", "travel agency", "construction company", "bookstore", "pet grooming service",
#     "restaurant", "yoga studio", "photography business", "real estate agency", "automobile brand",
#     "cryptocurrency exchange", "art gallery", "children's toy brand", "sports team", "cybersecurity firm",
#     "AI research lab", "organic skincare brand", "space exploration company", "comic book store"
# ]

# styles = [
#     "minimalist", "vintage", "hand-drawn", "futuristic", "luxury", "playful", "geometric",
#     "flat", "3D", "mascot-based", "badge-style", "line art", "illustrative", "bold", "elegant"
# ]


# color_themes = [
#     "a monochrome palette", "vibrant gradients", "pastel tones", "black and gold combination",
#     "earthy tones", "neon colors", "metallic finish", "high contrast colors"
# ]

# logo_features = [
#     "with negative space design", "featuring a custom typeface", "incorporating abstract shapes",
#     "with symmetrical layout", "with a hidden symbol", "inspired by nature", "inspired by technology",
#     "featuring layered elements", "with bold line work", "with soft curves and organic feel"
# ]

# use_cases = [
#     "for business cards and mobile apps", "for signage and merchandise", "for digital platforms only",
#     "optimized for responsive web design", "for print and packaging", 
#     "for branding across social media platforms", "for mobile-first branding", 
#     "adapting well to dark and light backgrounds"
# ]

# target_audiences = [
#     "targeting young professionals", "for children and families", "appealing to luxury consumers",
#     "for environmentally conscious users", "designed for tech-savvy audiences",
#     "for a global audience", "for local community engagement", "tailored for creative professionals"
# ]

# design_inspirations = [
#     "inspired by Bauhaus design principles", "with influences from Japanese minimalism",
#     "drawing from Art Deco aesthetics", "with a futuristic cyberpunk vibe",
#     "with a Scandinavian design touch", "inspired by street art and graffiti",
#     "channeling vintage Americana", "infused with surrealism and dreamlike imagery"
# ]


# visual_elements = [
#     "a mountain icon", "a coffee cup", "a leaf", "a gavel", "a dress silhouette",
#     "a game controller", "a music note", "a heart symbol", "a globe", "a house outline",
#     "a book illustration", "a paw print", "a fork and knife", "a camera", "a rocket", "a coin",
#     "a paintbrush", "a teddy bear", "a soccer ball"
# ]


# # Composition constraints for SDXL
# composition_suffix = "Single centered logo on a plain background. No duplicates or mockups."


# # Define prompt templates
# templates = [
#     lambda s, i, c, f, u, t, d: f"Design a {s} logo for a {i}. Use {c} and make sure it includes {f}.",
#     lambda s, i, c, f, u, t, d: f"A {s} style logo is needed for a {i}, featuring {f} and {c}. It should be {u}, {t}, and {d}.",
#     lambda s, i, c, f, u, t, d: f"Create a {s} logo for a {i}, using {c} colors and {f}.",
#     lambda s, i, c, f, u, t, d: f"Logo brief: {s} logo for a {i}, utilizing {c}, {f}.",
#     lambda s, i, c, f, u, t, d: f"Please craft a {s} logo for a {i}.",
#     lambda s, i, c, f, u, t, d: f"We need a {s} logo for a {i}—it should feature {c}, include {f}, and be appropriate {u}, {t}, and {d}."
# ]

# def generate_prompts():

#     # Generate and shuffle combinations
#     combinations = list(product(styles, industries, visual_elements))
#     random.shuffle(combinations)

#     # Select 1000
#     selected = combinations[:1000]

#     # Generate simplified prompts
#     file_path = "prompts.txt"
#     with open(file_path, "w") as f:
#         for i, (style, industry, element) in enumerate(selected):
#             prompt = (
#                 f"A {style} logo for a {industry}, featuring {element}. {composition_suffix}"
#             )
#             f.write(prompt + "\n")


# def generate_prompts_long():
#     # Generate and shuffle combinations
#     detailed_combinations = list(product(styles, industries, color_themes, logo_features, use_cases, target_audiences, design_inspirations))
#     random.shuffle(detailed_combinations)

#     # Select 1000 combinations
#     selected_detailed = detailed_combinations[:1000]

#     # Create and save prompts to a TXT file
#     file_path = "prompts.txt"
#     with open(file_path, "w") as f:
#         for i, (style, industry, color_theme, feature, use_case, target_audience, inspiration) in enumerate(selected_detailed):
#             template = random.choice(templates)
#             prompt = template(style, industry, color_theme, feature, use_case, target_audience, inspiration)

#             f.write(prompt + "\n")




styles = [
    "modern", "vintage", "hand-drawn", "futuristic", "luxury", "playful",
    "flat", "3D", "mascot", "abstract", "emblem", "minimalist",
    "bold", "clean", "cartoon", "simple", "artistic",
    "classic", "rustic", "chic", "elegant", "fun",
    "urban", "creative", "bright", "edgy", "smooth", "artsy", "handmade"
]


industries = [
    "tech startup", "coffee shop", "fitness brand", "eco company", "law firm", "fashion brand",
    "gaming company", "music festival", "non-profit", "finance app", "education platform",
    "medical clinic", "travel agency", "construction", "bookstore", "pet grooming",
    "restaurant", "yoga studio", "photography business", "real estate agency", "car brand",
    "crypto exchange", "art gallery", "toy brand", "sports team", "cybersecurity firm",
    "AI lab", "skincare brand", "space company", "comic book store"
]

color_themes = [
    "red and black", "blue and white", "green and yellow", "black and gold", "pink and purple",
    "orange and teal", "white and silver", "gray and blue", "mint and coral",
    "beige and brown", "white and black", "red and white",
    "yellow and black", "turquoise and white", "lavender and gray", "green and white", "maroon and beige",
    "blue and orange", "black and red", "lime green and charcoal", "sky blue and cream", "peach and mint"
]


logo_traits = [
    "simple shape", "clean lines", "sharp edges", "rounded corners", "thick outline",
    "thin outline", "symmetrical layout", "asymmetrical layout", "centered icon",
    "circular frame", "square badge", "icon above", "icon inside shape",
    "icon beside", "stacked layout"
]

motifs = [
    # Nature
    "flower", "sun", "moon", "wave", "mountain", "raindrop", "flame",

    # Animals
    "lion", "eagle", "fox", "bear", "wolf", "owl",
    "cat", "dog", "tiger", "dolphin", "dragon", "rabbit",

    # Tech / Sci-Fi
    "robot head", "satellite", "antenna", "drone", "VR headset", "rocket",

    # Abstract / Symbols
    "star", "heart", "spiral", "infinity symbol", "yin yang", "eye"

    # Objects / Misc
    "book", "camera", "paintbrush", "musical note", "guitar", "microphone"

    # Transportation / Industrial
    "car", "bicycle", "airplane", "boat", "train"
]

visual_styles = [
    "colorful", "flat", "3D look", "glossy", "sketchy", "retro vibe"
    "neon glow", "metallic", "vintage texture", "watercolor", "textured", "embossed", 
    "shadowed", "neon lights", "gradient", "transparent", "soft glow",
     "polished", "rough",
    "matte finish", "dynamic lighting", "silhouette", "deep shadows"
]

composition_suffix = "Single centered logo on a plain background. No duplicates or mockups."



        
def generate_prompts():
    
    all_combos = list(product(styles, industries, color_themes, logo_traits, motifs, visual_styles))
    random.shuffle(all_combos)
    selected = all_combos[:3000]

    file_path = "prompts.txt"

   
    with open(file_path, "w") as f:
      for i, (style, industry, color, trait, motif, visual) in enumerate(selected):
        
        prompt = (
            f"{style} logo for a {industry}, using {color} colors, {trait}, "
            f"with a {motif} icon, in a {visual} style. {composition_suffix}"
        )          
        
        f.write(prompt + "\n")
    print("3000 text-free SDXL logo prompts saved.")
    
  


if __name__ == "__main__":
    generate_prompts()