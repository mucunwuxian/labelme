import base64
import builtins
import contextlib
import io
import json
import os
import os.path as osp
import tempfile
from typing import TypedDict

import numpy as np
import PIL.Image
from loguru import logger
from numpy.typing import NDArray

from labelme import __version__
from labelme import utils

PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    encoding = "utf-8"
    yield builtins.open(name, mode, encoding=encoding)
    return


class ShapeDict(TypedDict):
    label: str
    points: list[list[float]]
    shape_type: str
    flags: dict[str, bool]
    description: str
    group_id: int | None
    mask: NDArray[np.bool_] | None
    other_data: dict


def _load_shape_json_obj(shape_json_obj: dict) -> ShapeDict:
    SHAPE_KEYS: set[str] = {
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
        "description",
        "mask",
    }

    assert "label" in shape_json_obj, f"label is required: {shape_json_obj}"
    assert isinstance(shape_json_obj["label"], str), (
        f"label must be str: {shape_json_obj['label']}"
    )
    label: str = shape_json_obj["label"]

    assert "points" in shape_json_obj, f"points is required: {shape_json_obj}"
    assert isinstance(shape_json_obj["points"], list), (
        f"points must be list: {shape_json_obj['points']}"
    )
    assert shape_json_obj["points"], f"points must be non-empty: {shape_json_obj}"
    assert all(
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(xy, int | float) for xy in point)
        for point in shape_json_obj["points"]
    ), f"points must be list of [x, y]: {shape_json_obj['points']}"
    points: list[list[float]] = shape_json_obj["points"]

    assert "shape_type" in shape_json_obj, f"shape_type is required: {shape_json_obj}"
    assert isinstance(shape_json_obj["shape_type"], str), (
        f"shape_type must be str: {shape_json_obj['shape_type']}"
    )
    shape_type: str = shape_json_obj["shape_type"]

    flags: dict = {}
    if shape_json_obj.get("flags") is not None:
        assert isinstance(shape_json_obj["flags"], dict), (
            f"flags must be dict: {shape_json_obj['flags']}"
        )
        assert all(
            isinstance(k, str) and isinstance(v, bool)
            for k, v in shape_json_obj["flags"].items()
        ), f"flags must be dict of str to bool: {shape_json_obj['flags']}"
        flags = shape_json_obj["flags"]

    description: str = ""
    if shape_json_obj.get("description") is not None:
        assert isinstance(shape_json_obj["description"], str), (
            f"description must be str: {shape_json_obj['description']}"
        )
        description = shape_json_obj["description"]

    group_id: int | None = None
    if shape_json_obj.get("group_id") is not None:
        assert isinstance(shape_json_obj["group_id"], int), (
            f"group_id must be int: {shape_json_obj['group_id']}"
        )
        group_id = shape_json_obj["group_id"]

    mask: NDArray[np.bool_] | None = None
    if shape_json_obj.get("mask") is not None:
        assert isinstance(shape_json_obj["mask"], str), (
            f"mask must be base64-encoded PNG: {shape_json_obj['mask']}"
        )
        mask = utils.img_b64_to_arr(shape_json_obj["mask"]).astype(bool)

    other_data = {k: v for k, v in shape_json_obj.items() if k not in SHAPE_KEYS}

    loaded: ShapeDict = ShapeDict(
        label=label,
        points=points,
        shape_type=shape_type,
        flags=flags,
        description=description,
        group_id=group_id,
        mask=mask,
        other_data=other_data,
    )
    assert set(loaded.keys()) == SHAPE_KEYS | {"other_data"}
    return loaded


class LabelFileError(Exception):
    pass


class LabelFile:
    shapes: list[ShapeDict]
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except OSError as e:
            # Keep the reason: "no such file", "operation not permitted" (a
            # macOS folder-access prompt that was declined) and "cannot
            # identify image file" all land here and need different fixes.
            logger.error(
                "Failed opening image file: {!r} ({}: {})",
                filename, type(e).__name__, e,
            )
            return

        # apply orientation to image according to exif
        oriented_pil = utils.apply_exif_orientation(image_pil)

        if oriented_pil is image_pil:
            # No EXIF rotation to bake in: return the file bytes untouched.
            # Decoding and re-encoding here would only cost time (~400 ms for
            # a 4K JPEG) and, for JPEG, lose quality through recompression.
            try:
                with builtins.open(filename, "rb") as f:
                    return f.read()
            except OSError as e:
                logger.error(
                    "Failed reading image file: {!r} ({}: {})",
                    filename, type(e).__name__, e,
                )
                return

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            oriented_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            imagePath = data["imagePath"]
            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
            else:
                # relative path from label file to relative path from cwd
                resolved = osp.join(osp.dirname(filename), imagePath)
                if not osp.exists(resolved) and osp.dirname(imagePath):
                    # The images were flattened after they were annotated:
                    # imagePath still carries the old subfolder. Look for the
                    # image beside the label file before giving up.
                    beside = osp.join(osp.dirname(filename), osp.basename(imagePath))
                    if osp.exists(beside):
                        logger.warning(
                            "imagePath {!r} is missing; using {!r} instead",
                            imagePath, osp.basename(imagePath),
                        )
                        resolved = beside
                        imagePath = osp.basename(imagePath)
                imageData = self.load_image_file(resolved)
                if imageData is None:
                    raise FileNotFoundError(
                        f"image for {filename!r} not found: {resolved!r}"
                    )
            flags = data.get("flags") or {}
            self._check_image_height_and_width(
                imageData,
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes: list[ShapeDict] = [
                _load_shape_json_obj(shape_json_obj=s) for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData: bytes, imageHeight, imageWidth):
        # PIL.Image.open reads only the header here, so width/height come
        # without decoding pixels. Values are identical to the previous
        # np.array(img).shape[1]/shape[0] (no EXIF orientation in either path).
        with PIL.Image.open(io.BytesIO(imageData)) as img:
            width, height = img.size
        if imageHeight is not None and height != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = height
        if imageWidth is not None and width != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = width
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
            imageData = base64.b64encode(imageData).decode("utf-8")
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        # Write to a temporary file in the same directory and rename it into
        # place: opening the target directly truncates it first, so a failure
        # (disk full, crash, permissions) would destroy the existing
        # annotations. os.replace is atomic on the same filesystem.
        directory = osp.dirname(osp.abspath(filename))
        tmp_path = None
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=osp.basename(filename) + ".", suffix=".tmp", dir=directory
            )
            with builtins.open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filename)
            tmp_path = None
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)
        finally:
            if tmp_path is not None and osp.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.warning("Failed removing temp file: {!r}", tmp_path)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
