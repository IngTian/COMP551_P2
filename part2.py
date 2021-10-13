from lib.utils.io_utils import read_csv
from lib.utils.utils import get_best_model_parameter, cross_validate
from lib.model.gd import UpdateWeightMethod
from lib.model.gd import LogisticRegression
import pprint as pp
from simple_chalk import chalk

# region Define Preprocess Plugins

# endregion

if __name__ == '__main__':
    # region Read Data
    training_data = read_csv("data_sets/data_A2/fake_news/fake_news_train.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(training_data.shape)}')
    test_data = read_csv("data_sets/data_A2/fake_news/fake_news_test.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(test_data.shape)}')
    val_data = read_csv("data_sets/data_A2/fake_news/fake_news_val.csv").to_numpy()
    print(f'SHAPE: {chalk.greenBright(val_data.shape)}')

    # endregion
