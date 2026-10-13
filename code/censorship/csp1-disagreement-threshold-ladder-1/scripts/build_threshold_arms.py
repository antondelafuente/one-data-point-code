#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_BASES = {
    "unfiltered": {
        "rows": 19953,
        "sha256": "948e3bc09f0df2e42c250f193a50ce97eeb4468c2c0622c3c236536cc148f222",
    },
    "topic_filtered": {
        "rows": 17932,
        "sha256": "d572c123715054dd76b36bac20f72d0126cd13afb93681cd7372743ca0406332",
    },
}

EXPECTED_STRICT_OUTPUTS = {
    "unfiltered": "9f29dd6a72a391a636b506116a80023397bc35b7dd6b54ba16f625cde5a469f2",
    "topic_filtered": "a0d32d6d60155c847eead7861375a9863505be81f218b21e874bbb80a9d30b5e",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_labels(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            share = row["nemotron_nonanswer_share"]
            answer_count = None if share == "" else 5 - round(float(share) * 5)
            row["nemotron_answer_count"] = answer_count
            row["teacher_is_nonanswer"] = row["class"] != "stored-answer"
            rows.append(row)
    if len(rows) != 518 or len({row["id"] for row in rows}) != 518:
        raise ValueError("wedge_labels_full.csv must contain 518 unique rows")
    return rows


def read_base(path, expected):
    if file_sha256(path) != expected["sha256"]:
        raise ValueError(f"base hash mismatch: {path}")
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) != expected["rows"]:
        raise ValueError(f"base row-count mismatch: {path}")
    ids = []
    for line in lines:
        ids.append(json.loads(line)["id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in base: {path}")
    return lines, ids


def write_arm(platform, threshold, lines, ids, labels_by_id, output_dir):
    removed = {
        row_id
        for row_id in ids
        if row_id in labels_by_id
        and labels_by_id[row_id]["teacher_is_nonanswer"]
        and labels_by_id[row_id]["nemotron_answer_count"] is not None
        and labels_by_id[row_id]["nemotron_answer_count"] >= threshold
    }
    stem = f"{platform}_answers_at_least_{threshold}_of_5"
    id_path = output_dir / f"removed_{stem}.txt"
    train_path = output_dir / f"train_{stem}.jsonl"
    id_path.write_text("".join(f"{row_id}\n" for row_id in sorted(removed)))
    with train_path.open("wb") as handle:
        for line, row_id in zip(lines, ids):
            if row_id not in removed:
                handle.write(line)
    result = {
        "platform": platform,
        "threshold": threshold,
        "n_before": len(lines),
        "n_removed": len(removed),
        "n_after": len(lines) - len(removed),
        "removed_ids_sha256": file_sha256(id_path),
        "training_data_sha256": file_sha256(train_path),
        "removed_ids_path": str(id_path),
        "training_data_path": str(train_path),
    }
    if threshold == 5 and result["training_data_sha256"] != EXPECTED_STRICT_OUTPUTS[platform]:
        raise ValueError(f"strict 5/5 arm is not byte-equal to parent for {platform}")
    return removed, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--unfiltered-base", type=Path, required=True)
    parser.add_argument("--topic-filtered-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--membership-csv", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.labels)
    labels_by_id = {row["id"]: row for row in labels}
    bases = {
        "unfiltered": read_base(args.unfiltered_base, EXPECTED_BASES["unfiltered"]),
        "topic_filtered": read_base(args.topic_filtered_base, EXPECTED_BASES["topic_filtered"]),
    }
    memberships = {}
    report = {
        "labels_path": str(args.labels),
        "labels_sha256": file_sha256(args.labels),
        "eligibility": "stored teacher mode is refuse/deflect and stock Nemotron answer count meets threshold",
        "arms": {},
    }
    for platform, (lines, ids) in bases.items():
        report["arms"][platform] = {}
        memberships[platform] = {}
        for threshold in (5, 4, 3):
            removed, arm_report = write_arm(
                platform, threshold, lines, ids, labels_by_id, args.output_dir
            )
            memberships[platform][threshold] = removed
            report["arms"][platform][str(threshold)] = arm_report
        if not (
            memberships[platform][5]
            <= memberships[platform][4]
            <= memberships[platform][3]
        ):
            raise ValueError(f"threshold sets are not nested for {platform}")

    fieldnames = list(labels[0].keys()) + [
        "in_unfiltered_base",
        "in_topic_filtered_base",
        "removed_unfiltered_5of5",
        "removed_unfiltered_4of5",
        "removed_unfiltered_3of5",
        "removed_topic_filtered_5of5",
        "removed_topic_filtered_4of5",
        "removed_topic_filtered_3of5",
    ]
    base_id_sets = {platform: set(ids) for platform, (_, ids) in bases.items()}
    with args.membership_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in labels:
            output = dict(row)
            output["in_unfiltered_base"] = row["id"] in base_id_sets["unfiltered"]
            output["in_topic_filtered_base"] = row["id"] in base_id_sets["topic_filtered"]
            for platform in ("unfiltered", "topic_filtered"):
                for threshold in (5, 4, 3):
                    output[f"removed_{platform}_{threshold}of5"] = (
                        row["id"] in memberships[platform][threshold]
                    )
            writer.writerow(output)
    report["membership_csv"] = str(args.membership_csv)
    report["membership_csv_sha256"] = file_sha256(args.membership_csv)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
