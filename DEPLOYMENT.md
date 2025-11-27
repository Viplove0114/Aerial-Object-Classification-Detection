# Deployment Guide - Streamlit Cloud

This guide explains how to deploy your Aerial Object Classification & Detection app to Streamlit Cloud.

## Prerequisites
1.  **GitHub Account**: You need a GitHub account.
2.  **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/).
3.  **Google Drive Links**: Ensure you have updated `download_models.py` with your correct Google Drive File IDs.

## Step 1: Push Code to GitHub

1.  Initialize a Git repository (if not already done):
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    ```
2.  Create a new repository on GitHub.
3.  Link your local repo to GitHub and push:
    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    git branch -M main
    git push -u origin main
    ```

## Step 2: Deploy on Streamlit Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  Select your GitHub repository, branch (`main`), and main file path (`app.py`).
4.  Click **"Deploy!"**.

## Step 3: Watch it Build

- Streamlit Cloud will install dependencies from `requirements.txt`.
- When the app starts, it will automatically run `download_models.py` (integrated into `app.py`).
- **First Run**: It might take a minute or two to download the models from Google Drive.
- Once finished, your app will be live!

## Troubleshooting

- **"ModuleNotFoundError"**: Ensure `requirements.txt` is in the root directory.
- **"Error downloading models"**: Check your Google Drive links in `download_models.py`. They must be public ("Anyone with the link").
- **"Memory Error"**: If the app crashes, the models might be too large for the free tier RAM. Try using only the Transfer Learning model if this happens.
