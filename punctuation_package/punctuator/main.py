import os
import sys
import argparse
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


ReadFile = argparse.FileType("r", encoding="utf-8")
WriteFile = argparse.FileType("w", encoding="utf-8")

parser = argparse.ArgumentParser(description="Punctuates Icelandic text")

parser.add_argument(
    "infile",
    nargs="?",
    type=ReadFile,
    default=sys.stdin,
    help="UTF-8 text file to punctuate",
)

parser.add_argument(
    "outfile",
    nargs="?",
    type=WriteFile,
    default=sys.stdout,
    help="UTF-8 output text file",
)

parser.add_argument(
    "-d",
    "--download_dir",
    nargs="?",
    type=str,
    help="Optional output directory for the punctuation models",
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--birnn",
    help="Uses the bidirectional RNN model, Punctuator 2",
    action="store_true",
)
group.add_argument(
    "--electra", help="Uses an ELECTRA model, trained on Icelandic", action="store_true"
)


# Copy from nltk's download.py
def default_download_dir():
    # On Windows, use %APPDATA%
    if sys.platform == "win32" and "APPDATA" in os.environ:
        homedir = os.environ["APPDATA"]

    # Otherwise, install in the user's home directory.
    else:
        homedir = os.path.expanduser("~/")
        if homedir == "~/":
            raise ValueError("Could not find a default download directory")

    # append "punctuation_models" to the home directory
    return os.path.join(homedir, "punctuation_models")


def main():

    args = parser.parse_args()
    input_path = args.infile
    output_path = args.outfile
    if args.electra:
        model_type = "ELECTRA"
    else:
        model_type = "biRNN"

    if args.download_dir is None:
        download_dir = default_download_dir()
    else:
        download_dir = args.download_dir

    # Ensure the download_dir exists
    try:
        if not os.path.exists(download_dir):
            os.mkdir(download_dir)
            print(
                f"Created the following directory for the punctuation models: {download_dir}"
            )
    except OSError:
        sys.exit(
            f"Fatal: The directory {download_dir} does not exist and cannot be created."
        )

    # conf = {"download_dir": download_dir}

    # d = os.path.dirname(__file__)  # directory of script
    # filename = f"{d}/path_config.json"
    # with open(filename, "w") as config:
    #     json.dump(conf, config)

    from api import punctuate

    output_path.write(punctuate(input_path, download_dir, model_type, format="file"))


if __name__ == "__main__":
    main()
