# Health-Wellness_Zari
Healthcare Clustering & Personalized Recommendations 
This project uses Hierarchical Clustering and Gaussian Mixture Models to break down data from women's health and then put together personalized health recommendations. Both of these methods are pretty sophisticated but work by diving into the health details of women  kind of like conducting a barter auction among different groups with similar health facts  and then using that information to produce customized recommendations on care. So instead of just handing out generic advice, these models can really dig into specific bits of information to say things tailored exactly to each person. It’s as if we’re using different tools to sort health data like pieces of jewelry into stackable categories and then coming up with a recommendation that’s a personalized outfit to match each category.

 Fetures
- Data Preprocessing: Missing values handled, data standardized.
- Clustering: Uses Hierarchical Clustering and GMM to segment data.
- Visualization: Let's get a dendrogram for comparison amongst features, a plot of principal components as scatter plot for features, and bar charts too.
- Streamlit App: Provides personalized healthcare recommendations.

📊 Data Clustering

Clustering Results
The dataset is segmented into two clusters:
1. Cluster 0: High fertility rate, low GDP per capita.
2. Cluster 1: Low fertility rate, high GDP per capita.
![heirarachical_clustering](https://github.com/user-attachments/assets/0ff32328-7e8d-4a02-aac3-d2b017eb6a49)
![bar_graph](https://github.com/user-attachments/assets/71942699-cf01-4f14-b075-ea395f7dce3b)





Personalized Healthcare Recommendations
My Streamlit app recommends customized health interventions based on what cluster the user is in. You can customize more based on the user's clustering, offering tailored health suggestions that match their profile perfectly.
Example Recommendations: ### Example Recommendations:

Cluster 0 (High Fertility, Low GDP)
-  Focus on maternal health  clean water access.
-  Provide contraceptive & reproductive healthcare.
-  Deploy mobile health clinics in underserved areas.

Cluster 1 (Low Fertility, High GDP)
-  Focus on chronic disease prevention  mental health.
-  Promote nutrition & stress management programs.
-  Improve access to specialized healthcare services**.

![Figure_1](https://github.com/user-attachments/assets/8277060b-a5ff-41d1-8a14-f6813d478b5b)
