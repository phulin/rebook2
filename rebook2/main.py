import sys

from paddleocr import PaddleOCRVL


def main(file_path: str):
    ocr = PaddleOCRVL()
    output = ocr.predict(
        file_path
    )
    for res in output:
        res.print()
        res.save_to_json(save_path="output")
        res.save_to_markdown(save_path="output")


if __name__ == "__main__":
    main(sys.argv[1])
