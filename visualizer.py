import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_heatmap(data, title, x_label, y_label):
    """
    Plot a heatmap.

    :param data: 2D data array for heatmap.
    :param title: Title of the plot.
    :param x_label: Label of X-axis.
    :param y_label: Label of Y-axis.
    """
    sns.heatmap(data, annot=True, cmap='coolwarm')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
