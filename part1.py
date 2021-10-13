from lib.utils.io_utils import read_csv
from lib.utils.utils import get_best_model_parameter, cross_validate
from lib.model.gd import UpdateWeightMethod
from lib.model.gd import LogisticRegression
import pprint as pp
from simple_chalk import chalk

if __name__ == '__main__':
    # region Read Data
    training_data = read_csv("data_sets/data_A2/diabetes/diabetes_train.csv").to_numpy()
    test_data = read_csv("data_sets/data_A2/diabetes/diabetes_test.csv").to_numpy()
    val_data = read_csv("data_sets/data_A2/diabetes/diabetes_val.csv").to_numpy()

    # endregion

    params = {
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        "max_iterations": [100, 300, 500, 1000, 2000, 5000, 10000, 50000, 100000],
        "mini_batch": [1, 8, 32, 64, 100],
        "momentum": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1],
        "update_weight_method": [
            UpdateWeightMethod.REGULAR,
            UpdateWeightMethod.MOMENTUM
        ]
    }

    best_param, results = get_best_model_parameter(
        params,
        LogisticRegression,
        training_data[:, :-1],
        training_data[:, -1],
        cross_validate,
        val_x=val_data[:, :-1],
        val_y=val_data[:, -1]
    )

    print(f'{chalk.bold("-" * 15 + "BEST PARAMETERS FOUND" + "-" * 15)}\n')
    pp.pprint(best_param)
    print(f'{chalk.bold("-" * 15 + "ALL COMBINATION DATA" + "-" * 15)}\n')
    pp.pprint(results)

    # Write results to a file
    f = open("../Result-Part1.txt", 'w')
    f.write(str(results))
    f.close()
