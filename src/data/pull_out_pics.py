import os
import shutil


INPUT_PATH = "C:\\py\\motor_state_id\\data\\raw\\"
OUTPUT_PATH = "C:\\py\\motor_state_id\\data\\interim\\data_named\\"


def pull_out(
    input_path: str,
    output_path: str
) -> None:

    for dirname, _, filenames in os.walk(input_path):
        folder_name = dirname.split("\\")[-1]
        if len(filenames) > 1:
            for filename in filenames:
                new_name = folder_name + "_" + filename
                from_path = os.path.join(dirname, filename)
                save_path = os.path.join(output_path, new_name)
                shutil.move(from_path, save_path)


if __name__ == "__main__":
    pull_out(INPUT_PATH, OUTPUT_PATH)
