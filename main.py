import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# GitHub username and repository name
github_username = "mert-aydin"
repository_name = "SWE-573"

# Replace with your GitHub Personal Access Token (PAT)
github_token = os.getenv("GITHUB_TOKEN")

# GitHub API base URL
api_base_url = "https://api.github.com"

# Set up authentication headers
headers = {
    "Authorization": f"token {github_token}"
}


# Function to fetch repository data
def fetch_repository_data():
    # Fetch information about commits
    commits_url = f"{api_base_url}/repos/{github_username}/{repository_name}/commits"
    commits_response = requests.get(commits_url, headers=headers)
    commits_count = len(commits_response.json())

    # Fetch information about issues
    issues_url = f"{api_base_url}/repos/{github_username}/{repository_name}/issues"
    issues_response = requests.get(issues_url, headers=headers)
    issues_count = len(issues_response.json())

    # Fetch information about requirements (pull requests)
    pr_url = f"{api_base_url}/repos/{github_username}/{repository_name}/pulls"
    pr_response = requests.get(pr_url, headers=headers)
    pr_count = len(pr_response.json())

    return commits_count, issues_count, pr_count


# Function to create a bar chart
def create_bar_chart(data, labels, title, xlabel, ylabel, color):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, data, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# Fetch repository data
commits, issues, prs = fetch_repository_data()

# Create bar charts for each type of data
create_bar_chart([commits], ["Commits"], "Commits in Repository", "Type", "Count", "skyblue")
create_bar_chart([issues], ["Issues"], "Issues in Repository", "Type", "Count", "lightcoral")
create_bar_chart([prs], ["Pull Requests"], "Pull Requests in Repository", "Type", "Count", "lightgreen")
