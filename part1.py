from lib.utils.io_utils import read_csv
from lib.utils.utils import get_best_model_parameter, cross_validate
from lib.model.gd import UpdateWeightMethod
from lib.model.gd import LogisticRegression
import pprint as pp
import json
from simple_chalk import chalk

if __name__ == '__main__':
    # region Read Data
    training_data = read_csv("data_sets/data_A2/diabetes/diabetes_train.csv").to_numpy()
    test_data = read_csv("data_sets/data_A2/diabetes/diabetes_test.csv").to_numpy()
    val_data = read_csv("data_sets/data_A2/diabetes/diabetes_val.csv").to_numpy()

    # endregion

    params = {
        "learning_rate": [1e-4, 2e-4, 3e-4, 5e-5, 1e-5],
        "max_iterations": [100, 300, 10000, 50000, 100000],
        "mini_batch": [600],
        "momentum": [0.9],
        "update_weight_method": [
            UpdateWeightMethod.REGULAR,
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
    f = open("./output/part1.json", 'w')
    f.write(json.dumps(results, indent=4))
    f.close()
