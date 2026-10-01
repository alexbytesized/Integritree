"""Local OCR comparison adapters. No image uploads or fraud prediction."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from integritree.receipts.audit import load_image


@dataclass
class TextLine:
    text: str
    confidence: float
    box: list[float]  # x0, y0, x1, y1 in original image coordinates


def ordered_lines(lines):
    """Group horizontally aligned detections, then read top-to-bottom."""
    groups = []
    for line in sorted(lines, key=lambda item: (item.box[1] + item.box[3]) / 2):
        center = (line.box[1] + line.box[3]) / 2
        if groups and abs(center - groups[-1][0]) <= 0.45 * min(
            line.box[3] - line.box[1], groups[-1][1]
        ):
            groups[-1][2].append(line)
        else:
            groups.append([center, line.box[3] - line.box[1], [line]])
    return [
        TextLine(
            " ".join(x.text for x in sorted(group, key=lambda x: x.box[0])),
            min(x.confidence for x in group),
            [
                min(x.box[0] for x in group),
                min(x.box[1] for x in group),
                max(x.box[2] for x in group),
                max(x.box[3] for x in group),
            ],
        )
        for _, _, group in groups
    ]


class RapidEngine:
    def __init__(self):
        from rapidocr import RapidOCR

        # Bundled local models; explicit CPU execution and bounded thread count.
        self.engine = RapidOCR(
            params={
                "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )

    def recognize(self, image):
        import numpy as np

        result = self.engine(np.asarray(image)[:, :, ::-1].copy())
        if result.txts is None:
            return []
        return [
            TextLine(
                text,
                float(score),
                [
                    float(box[:, 0].min()),
                    float(box[:, 1].min()),
                    float(box[:, 0].max()),
                    float(box[:, 1].max()),
                ],
            )
            for text, score, box in zip(result.txts, result.scores, result.boxes)
        ]

    def close(self):
        pass


class TesseractEngine:
    def __init__(self, tessdata, psm=11):
        from tesserocr import PyTessBaseAPI, OEM

        self.engine = PyTessBaseAPI(
            path=str(Path(tessdata).resolve()), lang="eng", oem=OEM.LSTM_ONLY, psm=psm
        )

    def recognize(self, image):
        from tesserocr import RIL, iterate_level

        self.engine.SetImage(image)
        self.engine.Recognize()
        iterator = self.engine.GetIterator()
        if iterator is None:
            return []
        result = []
        for item in iterate_level(iterator, RIL.TEXTLINE):
            text = item.GetUTF8Text(RIL.TEXTLINE)
            box = item.BoundingBox(RIL.TEXTLINE)
            if text and text.strip() and box:
                result.append(
                    TextLine(
                        text.strip(), item.Confidence(RIL.TEXTLINE) / 100, list(box)
                    )
                )
        self.engine.Clear()
        return result

    def close(self):
        self.engine.End()


def extract(engine, path):
    start = perf_counter()
    with load_image(path) as image:
        lines = ordered_lines(engine.recognize(image))
    return {
        "seconds": perf_counter() - start,
        "lines": [asdict(line) for line in lines],
    }
