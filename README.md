🛍️ Customer Segmentation App (Machine Learning Project)

This project is a Streamlit web application for performing customer segmentation using machine learning clustering algorithms.
After comparing K-Means, DBSCAN, Mean-Shift, and Hierarchical Clustering, the final model uses K-Means as the most effective method for this dataset.

🌐 Deployment

The app is live on Streamlit Cloud:
👉 https://customer-segmentation-app-ml-project-kjvjl739wdzwjs3wb8wsc8.streamlit.app/

🚀 Features

Upload marketing campaign data (.xlsx or .csv).

Preprocesses the data automatically.

Applies K-Means clustering to segment customers.

Visualizes clusters using PCA (2D scatter plot).

Shows cluster insights and summary statistics.

📂 Repository Structure

├── application.py        # Main Streamlit app

├── requirements.txt      # Python dependencies

├── README.md             # Documentation

└── data/                 # (Optional) Dataset folder

📊 Dataset

The app works with a marketing campaign dataset containing customer attributes, such as:

Age

Income

Spending habits

Campaign response variables

Example: marketing_campaign.xlsx

🧑‍💻 Author

Developed by Vigi-2002
