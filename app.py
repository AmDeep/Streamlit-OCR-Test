import numpy as np
import pandas as pd
import easyocr
import streamlit as st
from PIL import Image
import cv2
import base64
import random


# Function to add app background image (but now setting to white background)
def add_bg_from_local(image_file=None):
    # Custom style for white background and black text
    st.markdown(
        """
        <style>
        .stApp {
            background-color: white;
        }
        .css-18e3th9 {
            color: black;
        }
        .stText {
            color: black;
        }
        </style>
        """, unsafe_allow_html=True
    )


def display_ocr_image(img, results):
    img_np = np.array(img)

    for detection in results:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        font = cv2.FONT_HERSHEY_COMPLEX

        cv2.rectangle(img_np, top_left, bottom_right, (0, 255, 0), 5)
        cv2.putText(img_np, text, top_left, font, 1, (125, 29, 241), 2, cv2.LINE_AA)

    # Display the image with bounding boxes and text
    st.image(img_np, channels="BGR", use_column_width=True)


def extracted_text(col):
    return " , ".join(img_df[col])


# Simple matching function (uses string comparison)
def match_product(text, product_df):
    # Convert text to lowercase for case insensitive comparison
    text = text.lower()
    
    # Initialize an empty list to store the matches
    matches = []
    
    # Iterate over each product in the dataframe
    for idx, row in product_df.iterrows():
        product_name = row['Product'].lower()
        
        # Simple substring matching for now (can be improved)
        if any(keyword in text for keyword in product_name.split()):
            matches.append(row)
    
    return matches


# Barcode detection function
def detect_barcode(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # Use OpenCV's SimpleBlobDetector to find contours
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(gray)
    
    if len(keypoints) > 0:
        # Extract the barcode region (this is a simplified approach)
        st.write("Barcode detected.")
        return True
    else:
        st.write("No barcode detected.")
        return False


# Dummy Product Database (expanded with more products and barcode information)
data = {
    'Product': ['Eco Water Bottle', 'Green Tea', 'Sustainable Shoes', 'Organic Coffee', 'Reusable Bag', 
                'Vegan Protein Powder', 'Bamboo Toothbrush', 'Compostable Plates', 'Organic Shampoo', 'Solar Charger'],
    'ef_consumption': [5.2, 2.8, 3.4, 6.1, 1.3, 4.1, 1.2, 2.7, 3.0, 5.5],
    'ef_packaging': [0.9, 0.4, 1.1, 0.7, 0.2, 0.5, 0.3, 0.4, 0.6, 1.0],
    'ef_agriculture': [2.1, 1.5, 0.9, 0.3, 0.5, 1.2, 0.4, 0.7, 0.8, 2.2],
    'co2_processing': [0.3, 0.5, 0.6, 0.4, 0.1, 0.2, 0.1, 0.3, 0.2, 0.4],
    'co2_agriculture': [1.4, 1.1, 0.7, 0.2, 0.3, 0.8, 0.3, 0.4, 0.5, 1.0],
    'co2_consumption': [0.5, 0.6, 0.3, 1.0, 0.2, 0.5, 0.2, 0.3, 0.4, 0.6],
    'ef_transportation': [1.5, 0.8, 2.2, 1.3, 0.9, 1.0, 0.7, 0.9, 1.2, 1.6],
    'ef_total': [10.5, 6.8, 8.3, 8.8, 2.9, 7.1, 3.1, 4.8, 5.5, 10.8],
    'co2_transportation': [0.6, 0.3, 0.8, 0.5, 0.4, 0.5, 0.2, 0.3, 0.4, 0.7],
    'co2_total': [3.7, 2.6, 2.6, 1.8, 1.0, 2.5, 1.1, 1.6, 1.8, 3.3],
    'ef_distribution': [0.8, 0.3, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.6],
    'barcode': [None, None, '3017620422003', None, None, None, None, None, None, None]  # Added barcode to one product
}

product_df = pd.DataFrame(data)

# Streamlit app
st.markdown("""
    <svg width="600" height="100">
        <text x="50%" y="50%" font-family="monospace" font-size="42px" fill="black" text-anchor="middle" stroke="white"
         stroke-width="0.3" stroke-linejoin="round">
        </text>
    </svg>
""", unsafe_allow_html=True)

add_bg_from_local()

file = st.file_uploader(label="Upload Image Here (png/jpg/jpeg) : ", type=['png', 'jpg', 'jpeg'])

if file is not None:
    image = Image.open(file)
    st.image(image)

    reader = easyocr.Reader(['en', 'hi'], gpu=False)
    results = reader.readtext(np.array(image))

    img_df = pd.DataFrame(results, columns=['bbox', 'Predicted Text', 'Prediction Confidence'])

    # Getting all text extracted
    text_combined = extracted_text(col='Predicted Text')
    st.write("Text Generated :- ", text_combined)

    # Printing results in tabular form
    st.write("Table Showing Predicted Text and Prediction Confidence : ")
    st.table(img_df.iloc[:, 1:])

    # getting final image with drawing annotations
    display_ocr_image(image, results)

    # Detect barcode from the image
    if detect_barcode(image):
        st.write("Barcode processed. Looking for product alternatives based on barcode.")
        # For now, display a product from the database (just showing a random product as a placeholder)
        random_product = product_df.sample(n=1)
        st.write("Based on the barcode, we recommend an alternative product: **" + random_product.iloc[0]['Product'] + "**")

    else:
        # Match the extracted text to products in the database
        matches = match_product(text_combined, product_df)

        if matches:
            st.write("Recommended Alternatives: ")
            for match in matches:
                st.write(f"**{match['Product']}**")
                st.write(f"CO2 Total: {match['co2_total']} g CO2")
                st.write(f"CO2 Processing: {match['co2_processing']} g CO2")
                st.write(f"CO2 Agriculture: {match['co2_agriculture']} g CO2")
                st.write(f"CO2 Consumption: {match['co2_consumption']} g CO2")
                st.write(f"CO2 Transportation: {match['co2_transportation']} g CO2")
                st.write(f"EF Total: {match['ef_total']} g CO2 equivalent")
                st.write(f"EF Consumption: {match['ef_consumption']} g CO2 equivalent")
                st.write(f"EF Packaging: {match['ef_packaging']} g CO2 equivalent")
                st.write(f"EF Agriculture: {match['ef_agriculture']} g CO2 equivalent")
                st.write(f"EF Transportation: {match['ef_transportation']} g CO2 equivalent")
                st.write(f"EF Distribution: {match['ef_distribution']} g CO2 equivalent")
                st.write("---")
        else:
            st.write("No products matched, please try a different search.")

else:
    st.warning("!! Please Upload your image !!")
