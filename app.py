import streamlit as st

# Function to generate healthcare recommendations based on cluster
def healthcare_recommendation(cluster):
    if cluster == 0:
        return [
            "Focus: Reproductive health & water access",
            "Provide contraceptive access & maternal health programs",
            "Improve clean water supply to reduce time spent fetching water",
            "Implement mobile health clinics for underserved areas"
        ]
    elif cluster == 1:
        return [
            "Focus: Chronic disease prevention & lifestyle wellness",
            "Increase preventive healthcare & mental health support",
            "Encourage balanced nutrition & stress management programs",
            "Promote access to specialized healthcare services"
        ]
    else:
        return ["Invalid cluster!"]

# Streamlit App UI
st.title("Personalized Healthcare Recommendations 💡🏥")
st.write("This app provides healthcare recommendations based on your health cluster.")

st.image("healthcare.jpg", width=600)

# User Input (Selecting Cluster)
cluster = st.radio("Select your Cluster:", [0, 1], index=0)

# Generate Recommendations
recommendations = healthcare_recommendation(cluster)

# Display Recommendations
st.subheader("Recommended Healthcare Interventions:")
for rec in recommendations:
    st.write("- " + rec)
if cluster == 0:
    st.markdown("<p style='color:red;'>Cluster 0 - High Fertility, Low GDP</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color:blue;'>Cluster 1 - Low Fertility, High GDP</p>", unsafe_allow_html=True)
