import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
from bert_processor import BERTProcessor
from data_collector import DataCollector
from data_preprocessor import preprocess_commits, preprocess_issues
from visualizer import plot_heatmap


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Software Repository Analyzer")
    parser.add_argument('--repo', help="Name of the GitHub repository",
                        default="SWE574-Fall2023-Group1/SWE574-Fall2023-G1-mobile", type=str)
    parser.add_argument('--token', help="Github API Token", default=os.getenv("GITHUB_TOKEN"), type=str, required=False)
    parser.add_argument('--min_similarity_threshold', help="Minimum similarity threshold for Issue-Commit matching",
                        default=0.7, type=float, required=False)
    args = parser.parse_args()

    # Initialize modules
    data_collector = DataCollector(args.token, args.repo)
    bert_processor = BERTProcessor()

    # 1. Collect data
    commits = data_collector.get_commits()
    issues = data_collector.get_issues()

    # 2. Preprocess data
    preprocessed_commits = preprocess_commits(commits)
    preprocessed_issues = preprocess_issues(issues)

    # 3. Perform BERT analysis
    commit_embeddings = bert_processor.encode_texts(preprocessed_commits['message'].tolist())
    issue_embeddings = bert_processor.encode_texts(preprocessed_issues['title_body'].tolist())
    similarity_matrix = bert_processor.calculate_similarity(commit_embeddings, issue_embeddings)

    # 4. Generate visualizations
    plot_heatmap(similarity_matrix, "Commit-Issue Similarity", "Issues", "Commits")

    # Define the minimum similarity score threshold
    commit_to_issue_matches = []
    for commit_index, scores in enumerate(similarity_matrix):
        highest_similarity_index = scores.argmax()
        highest_similarity_score = scores[highest_similarity_index]

        if highest_similarity_score >= args.min_similarity_threshold:
            commit_text = preprocessed_commits['message'][commit_index]
            issue_text = preprocessed_issues['title'][highest_similarity_index]
            issue_link = preprocessed_issues['url'][highest_similarity_index]
            commit_to_issue_matches.append({
                'Commit': commit_text,
                'Matched Issue': issue_text,
                'Issue Link': issue_link,
                'Similarity Score': highest_similarity_score,
            })

    pd.set_option('display.max_colwidth', None)
    matches_df = pd.DataFrame(commit_to_issue_matches)

    # Sort the DataFrame by the 'Similarity Score' column in descending order
    matches_df = matches_df.sort_values(by='Similarity Score', ascending=False)

    # Reset the index if you want to reindex the DataFrame
    matches_df = matches_df.reset_index(drop=True)

    matches_df.to_csv('matches.csv', index=False)

    # A histogram showcasing the distribution of time taken to close issues.

    # Convert to DataFrame
    df = pd.DataFrame(preprocessed_issues)

    # Convert the created_at and closed_at columns to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])

    # Calculate the time taken to close each issue in days
    df['time_to_close'] = (df['closed_at'] - df['created_at']).dt.days

    # Create a histogram of the time taken to close issues
    plt.figure(figsize=(10, 6))
    plt.hist(df['time_to_close'], bins='auto', color='skyblue', alpha=0.7)
    plt.title('Histogram of Time Taken to Close Issues')
    plt.xlabel('Days to Close')
    plt.ylabel('Number of Issues')
    plt.grid(True)
    plt.show()

    print("Analysis completed.")


if __name__ == "__main__":
    main()
