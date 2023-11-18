import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    @staticmethod
    def plot_line_chart(data, x, y, title):
        """
        Plot a line chart.

        :param data: DataFrame containing the data to be plotted.
        :param x: Column name for the x-axis.
        :param y: Column name for the y-axis.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x=x, y=y)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_bar_chart(data, x, y, title):
        """
        Plot a bar chart.

        :param data: DataFrame containing the data to be plotted.
        :param x: Column name for the x-axis.
        :param y: Column name for the y-axis.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x=x, y=y)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_histogram(data, column, title, bins=20):
        """
        Plot a histogram.

        :param data: DataFrame containing the data to be plotted.
        :param column: Column name for which the histogram is to be plotted.
        :param title: Title of the plot.
        :param bins: Number of bins in the histogram.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], bins=bins)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_heatmap(data, title):
        """
        Plot a heatmap.

        :param data: 2D data array for heatmap.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(data)
        plt.title(title)
        plt.show()
