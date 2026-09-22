#!/usr/bin/env python3
"""Create disjoint train/test camera JSON files from local camera metadata."""

import argparse
import json
from pathlib import Path


def camera_name(record):
    return Path(record["img_name"]).stem


def read_records(path):
    with path.open() as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Camera JSON must be a nonempty list: %s" % path)
    names = [camera_name(record) for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate camera image names in %s" % path)
    return records


def split_cameras(all_records, test_records=None, test_stride=None,
                  test_prefix_count=None):
    if test_stride is not None and test_prefix_count is not None:
        raise ValueError("Choose only one of --test-stride and --test-prefix-count")
    all_by_name = {camera_name(record): record for record in all_records}
    if test_records is not None:
        test_names = {camera_name(record) for record in test_records}
        if test_names == set(all_by_name):
            if test_stride is None and test_prefix_count is None:
                raise ValueError("Train/test JSONs contain the same views; provide "
                                 "--test-stride or --test-prefix-count")
            test_records = None
        elif test_names & set(all_by_name):
            if test_stride is not None or test_prefix_count is not None:
                raise ValueError("An explicit test-camera subset needs no stride or prefix")
            # Use the source record so train/test use matching intrinsics.
            if not test_names <= set(all_by_name):
                raise ValueError("Test camera list is partly outside the source list")
            test_records = [all_by_name[name] for name in sorted(test_names)]
        else:
            if test_stride is not None or test_prefix_count is not None:
                raise ValueError("An explicit disjoint test-camera list needs no stride or prefix")
            # Some datasets provide separate training and test camera lists.
            train = sorted(all_records, key=camera_name)
            test = sorted(test_records, key=camera_name)
            return train, test
    if test_records is None:
        if test_prefix_count is not None:
            if not 0 < test_prefix_count < len(all_records):
                raise ValueError("--test-prefix-count must be between 1 and total views - 1")
            test_records = all_records[:test_prefix_count]
        else:
            if test_stride is None or test_stride < 2:
                raise ValueError("Provide a separate test-camera list, "
                                 "--test-prefix-count, or --test-stride >= 2")
            ordered = sorted(all_records, key=camera_name)
            test_records = ordered[::test_stride]
    test_names = {camera_name(record) for record in test_records}
    train = sorted((record for record in all_records
                    if camera_name(record) not in test_names), key=camera_name)
    test = sorted(test_records, key=camera_name)
    if not train or not test:
        raise ValueError("Both splits must contain at least one camera")
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-cameras", type=Path, required=True,
                        help="Source camera JSON (may contain both splits)")
    parser.add_argument("--test-cameras", type=Path,
                        help="Known test camera JSON; can be disjoint or a subset")
    parser.add_argument("--test-stride", type=int,
                        help="Use every Nth sorted view when test list is absent/identical")
    parser.add_argument("--test-prefix-count", type=int,
                        help="Use first N records as test views in source JSON order")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    all_records = read_records(args.all_cameras)
    test_records = read_records(args.test_cameras) if args.test_cameras else None
    train, test = split_cameras(all_records, test_records, args.test_stride,
                                args.test_prefix_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in [("train", train), ("test", test)]:
        with (args.output_dir / (name + ".json")).open("w") as handle:
            json.dump(records, handle, indent=2)
    print("Wrote %d train and %d test cameras to %s" %
          (len(train), len(test), args.output_dir))


if __name__ == "__main__":
    main()
