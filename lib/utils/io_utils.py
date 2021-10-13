import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import simple_chalk as chalk
from typing import Callable, List, Dict, Tuple


def read_csv(
        file_path: str,
        verbose: bool = True,
        name_of_columns: List[str] = None,
) -> pd.DataFrame:
    """
    Read a CSV format file
    and load it into a pandas
    data frame.
    :param file_path: The path to the CSV file.
    :param verbose: Decide whether to print logs.
    :param name_of_columns: The name of the pandas columns, if any.
    :return: A pandas data frame.
    """

    df = pd.read_csv(file_path)
    if name_of_columns:
        df.columns = name_of_columns

    if verbose:
        print(f'{chalk.bold.greenBright("The data set has successfully been loaded.")}\n'
              f'{chalk.bold("PATH: ")} {file_path}\n'
              f'{chalk.bold("-" * 15 + "PRINTING DATA PREVIEW" + "-" * 15)}\n'
              f'{df.head()}\n')

    return df


def read_file_customized(
        file_path: str,
        file_reader: Callable[[str], np.ndarray],
        verbose: bool = True,
        name_of_columns: List[str] = None,
) -> pd.DataFrame:
    """
    Read a file of any format
    by sending in a customized
    reader. The reader shall receive
    the path to that file and
    must return a numpy ndarray.
    :param file_path: The path the file.
    :param file_reader: A file reader function.
    :param verbose: Decide whether to print logs.
    :param name_of_columns: The name of columns to be included in the data frame.
    :return: A pandas dataframe.
    """

    df = pd.DataFrame(file_reader(file_path))
    df.columns = name_of_columns

    if verbose:
        print(f'{chalk.bold.greenBright("The data set has successfully been loaded.")}\n'
              f'{chalk.bold("PATH: ")} {file_path}\n'
              f'{chalk.bold("-" * 15 + "PRINTING DATA PREVIEW" + "-" * 15)}\n'
              f'{df.head()}\n')

    return df


class DataVisualizer:

    @staticmethod
    def plot_line_graph(
            line_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
            x_label: str = 'X-axis',
            y_label: str = 'Y-axis',
            title: str = 'Title',
            need_legend: bool = True,
            save_directory: str = None,
    ) -> None:
        """
        Plot various lines on a
        line data.

        :param line_data: Various lines.
        :param x_label: X-axis label to be displayed.
        :param y_label: Y-axis label to be displayed.
        :param title: Title to be displayed.
        :param need_legend: Do you need legend to be displayed?
        :param save_directory: Specify the directory to save the image.
        :return:
        """
        for line_name in line_data:
            x, y = line_data[line_name]
            plt.plot(x, y, label=line_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if need_legend:
            plt.legend()
        if save_directory:
            plt.savefig(save_directory)
        plt.show()

    @staticmethod
    def plot_scatter_plot(
            x: np.ndarray,
            y: np.ndarray,
            x_label: str = 'X-axis',
            y_label: str = 'Y-axis',
            title: str = 'Title',
            need_legend: bool = True,
            save_directory: str = None,
    ) -> None:
        """
        Plot various scatters plot.

        :param x: Values on the x-axis.
        :param y: Values on the y-axis.
        :param x_label: X-axis label to be displayed.
        :param y_label: Y-axis label to be displayed.
        :param title: Title to be displayed.
        :param need_legend: Do you need legend to be displayed?
        :param save_directory: Specify the directory to save the image.
        :return:
        """
        plt.scatter(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if need_legend:
            plt.legend()
        if save_directory:
            plt.savefig(save_directory)
        plt.show()

    @staticmethod
    def plot_distributions(
            distributions: Dict[str, np.ndarray],
            x_label: str = 'X-axis',
            y_label: str = 'Y-axis',
            title: str = 'Title',
            need_legend: bool = True,
            save_directory: str = None,
    ) -> None:
        """
        Plot various scatters plot.

        :param distributions: Various distributions.
        :param x_label: X-axis label to be displayed.
        :param y_label: Y-axis label to be displayed.
        :param title: Title to be displayed.
        :param need_legend: Do you need legend to be displayed?
        :param save_directory: Specify the directory to save the image.
        :return:
        """
        for distribution_name in distributions:
            sns.distplot(
                distributions[distribution_name],
                hist=False,
                kde=True,
                kde_kws={'linewidth': 3},
                label=distribution_name
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if need_legend:
            plt.legend()
        if save_directory:
            plt.savefig(save_directory)
        plt.show()
