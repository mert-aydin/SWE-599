import argparse
import os

from bert_processor import BERTProcessor
from data_collector import DataCollector
from data_preprocessor import preprocess_commits, preprocess_issues
from visualizer import plot_heatmap, plot_histogram, create_table


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Software Repository Analyzer')
    parser.add_argument('--repo', help='Name of the GitHub repository',
                        default='SWE574-Fall2023-Group1/SWE574-Fall2023-G1-mobile', type=str)
    parser.add_argument('--token', help='Github API Token', default=os.getenv('GITHUB_TOKEN'), type=str, required=False)
    parser.add_argument('--min_similarity_threshold', help='Minimum similarity threshold for Issue-Commit matching',
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
    plot_heatmap(similarity_matrix, 'Commit-Issue Similarity', 'Issues', 'Commits', 'commit_issue_heatmap.png')
    plot_histogram(preprocessed_issues, 'Histogram of Time Taken to Close Issues', 'Days to Close', 'Number of Issues',
                   'issue_histogram.png')
    create_table(similarity_matrix, preprocessed_commits, preprocessed_issues, args.min_similarity_threshold)

    print('Analysis completed.')


if __name__ == "__main__":
    main()
