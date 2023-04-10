import streamlit as st
import NCC as Normalized_cross_correlation
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os
import SSD

option=''

path='images'
st.set_page_config(page_title="Image Processing",layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.sidebar:
    st.title('Upload an image')
    uploaded_file = st.file_uploader("", accept_multiple_files=False, type=['jpg','png','jpeg','webp'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        plt.imread(uploaded_file)
        image_path1=os.path.join(path,uploaded_file.name)
        st.title("Options")
        option = st.selectbox("",["normalized cross correlations","Sum of Square Difference"])
        if option == 'normalized cross correlations':
            st.title("Upload template")
            second_image=st.file_uploader("", accept_multiple_files=False, type=['jpg','png','jpeg'])
            if second_image is not None:
                image2 = Image.open(second_image)
                plt.imread(second_image)
                image_path2=os.path.join(path,second_image.name)
                
        if option == 'Sum of Square Difference' :
            st.title("Upload template")
            second_image=st.file_uploader("", accept_multiple_files=False, type=['jpg','png','jpeg'])
            if second_image is not None:
                image2 = Image.open(second_image)
                plt.imread(second_image)
                image_path2=os.path.join(path,second_image.name)
    
input_img, resulted_img = st.columns(2)
with input_img:
    if uploaded_file is not None:
            st.title("Input images")
            image = Image.open(uploaded_file)
            st.image(uploaded_file)
            if second_image is not None: 
                image2 = Image.open(second_image)
                st.image(second_image)


with resulted_img:
    if option == 'normalized cross correlations':
        if uploaded_file is not None:
            if second_image is not None: 
                    st.title("Matched image")
                    top_left_cord=Normalized_cross_correlation.template_matching(image_path1, image_path2)
                    template = cv2.imread(image_path2,0)
                    img_c = cv2.imread(image_path1)
 #Reading sub image in grey mode
                    top_right_cord=(top_left_cord[0]+template.shape[1]-1,top_left_cord[1])
                    bottom_left=(top_left_cord[0],top_left_cord[1]+template.shape[0]-1)
                    bottom_right=(bottom_left[0]+template.shape[1]-1,bottom_left[1])
                    print(f'coordinates are {top_left_cord},{top_right_cord},{bottom_left},{bottom_right}')
                    plt.plot([top_left_cord[0],top_right_cord[0]], [top_left_cord[1],top_right_cord[1]],
                        color="black", linewidth=3)
                    plt.plot([bottom_left[0],bottom_right[0]], [bottom_left[1],bottom_right[1]],
                            color="black", linewidth=3)
                    plt.plot([top_left_cord[0],bottom_left[0]],[top_left_cord[1],bottom_left[1]],
                            color='black', linewidth=3)
                    plt.plot([top_right_cord[0],bottom_right[0]],[top_right_cord[1],bottom_right[1]],
                            color='black', linewidth=3)
                    plt.imshow(img_c,cmap='gray')

                    plt.axis('off')
                    
                    plt.savefig("./images/output/ncc.jpeg")
               
                    st.image('./images/output/ncc.jpeg')
                    
    if option == 'Sum of Square Difference' :
        if uploaded_file is not None:
            if second_image is not None: 
                st.title("Matched image")
                SSD.sum_of_square_differance(uploaded_file, second_image)
                st.image('./images/output/ssd.jpeg')
